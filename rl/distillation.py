"""学生访问状态蒸馏：稀疏帧库、按环境顺序BPTT和reset掩码。"""

import torch
import torch.nn.functional as F

from env.gait import flat_terrain_mask


@torch.no_grad()
def collect_sequences(task, visual, teacher, student, steps, beta, action_noise_std=0., held_terrain_guide=False,
                      flat_reference_visual=None, flat_weight=1.):
    n, device = task.config.num_envs, task.device
    # 图像仅在实际更新时入库；策略步只保存两帧索引，避免50Hz重复存整幅图。
    frames = [visual.pair[:, 0].clone(), visual.pair[:, 1].clone()]
    previous = torch.arange(n, device=device)
    current = previous+n
    frame_count = 2*n
    initial_hidden = visual.hidden_before.clone()
    tick = torch.ones(n, dtype=torch.bool, device=device)
    reset = torch.zeros_like(tick)
    rows = []
    held_teacher_z = torch.zeros(n, 64, device=device) if held_terrain_guide else None
    successes, episodes = 0, 0
    flat_samples=0
    for _ in range(steps):
        states, critic = task.inputs()
        if held_terrain_guide:
            # 标签与实际采图/reset时序一致，冻结学生控制网络生成特权引导动作。
            fresh_z = teacher.terrain_encoder(states[:, 495:])
            if flat_reference_visual is not None:
                # 平地保留源视觉特征；复杂地形才对齐教师高度编码。
                local_x=task.robot.data.root_link_pos_w[:,0]-task.env.scene.env_origins[:,0]+task.config.terrain.spawn_x
                is_flat=flat_terrain_mask(task, local_x)
                fresh_z=torch.where(is_flat[:,None], flat_reference_visual.latent, fresh_z)
                flat_samples+=int(is_flat.sum())
            held_teacher_z = torch.where(tick[:, None], fresh_z, held_teacher_z)
            teacher_z = held_teacher_z
            teacher_mu = student(torch.cat([states[:, :495], teacher_z], -1))[0]
        else:
            teacher_mu = teacher(states)[0]
            teacher_z = teacher.terrain_encoder(states[:, 495:])
        student_states = torch.cat([states[:, :495], visual.latent], -1)
        student_mu = student(student_states)[0]
        rows.append(dict(proprio=states[:, :45].clone(), history=states[:, 45:495].clone(),
                         action=teacher_mu.clone(), terrain=teacher_z.clone(),
                         velocity=critic[:, 45:48].clone(), tick=tick.clone(), reset=reset.clone(),
                         frame_ids=torch.stack([previous, current], -1)))
        if flat_reference_visual is not None:
            rows[-1]['weights']=1+(flat_weight-1)*is_flat.float()
        action = beta*teacher_mu+(1-beta)*student_mu
        if action_noise_std > 0:
            # 扩展落足相位附近的访问状态；监督标签仍为当前状态的教师均值动作。
            action = action+action_noise_std*torch.randn_like(action)
        reward, terminated, truncated, info = task.step(action)
        updated = visual.update()
        if flat_reference_visual is not None:
            flat_reference_visual.update()
        tick = torch.full_like(tick, updated)
        if updated:
            frames.append(visual.pair[:, 1].clone())
            previous, current = current, torch.arange(n, device=device)+frame_count
            frame_count += n
        reset = terminated | truncated
        ids = reset.nonzero(as_tuple=False).flatten()
        if len(ids):
            episodes += len(ids)
            successes += info['success'][ids].sum().item()
            task.reset(ids, info['success'][ids])
            visual.reset(ids)
            if flat_reference_visual is not None:
                flat_reference_visual.reset(ids)
            frames.append(visual.pair[ids, 1].clone())
            fresh = torch.arange(len(ids), device=device)+frame_count
            previous[ids], current[ids] = fresh, fresh
            frame_count += len(ids)
            tick[ids] = True
    batch = {key:torch.stack([row[key] for row in rows]) for key in rows[0]}
    batch['frames'], batch['initial_hidden'] = torch.cat(frames), initial_hidden
    sampling=dict(episodes=episodes, successes=successes, stored_depth_frames=frame_count)
    if flat_reference_visual is not None:
        sampling['flat_reference_fraction']=flat_samples/(steps*n)
    return batch, sampling


def merge_sequence_batches(batches):
    """合并独立任务的时序样本，修正稀疏帧索引，保留各世界reset和GRU状态。"""
    keys=set(batches[0])-{'frames','initial_hidden','frame_ids'}
    merged={key:torch.cat([batch[key] for batch in batches],1) for key in keys}
    indices=[]
    offset=0
    for batch in batches:
        indices.append(batch['frame_ids']+offset)
        offset+=len(batch['frames'])
    merged['frame_ids']=torch.cat(indices,1)
    merged['frames']=torch.cat([batch['frames'] for batch in batches])
    merged['initial_hidden']=torch.cat([batch['initial_hidden'] for batch in batches])
    return merged


def update_sequences(student, optimizer, batch, config):
    p, history = batch['proprio'], batch['history']
    burn = config.burn_in*5
    totals = torch.zeros(3, device=p.device)
    count = 0
    for ids in torch.randperm(p.shape[1], device=p.device).split(config.sequence_batch):
        hidden = batch['initial_hidden'][ids].detach()
        latent = torch.zeros(len(ids), 64, device=p.device)
        latents = []
        for t in range(len(p)):
            # burn-in用当前网络重新计算上下文；reset切断跨回合记忆与梯度。
            with torch.set_grad_enabled(t >= burn):
                hidden = hidden*(~batch['reset'][t, ids])[:, None]
                fresh = batch['tick'][t, ids].nonzero(as_tuple=False).flatten()
                if len(fresh):
                    images = batch['frames'][batch['frame_ids'][t, ids[fresh]]]
                    z, h = student.visual_encoder(images, p[t, ids[fresh]], hidden[fresh])
                    hidden = hidden.index_copy(0, fresh, h)
                    latent = latent.index_copy(0, fresh, z)
            if t >= burn:
                latents.append(latent)
        z = torch.stack(latents)
        states = torch.cat([p[burn:, ids], history[burn:, ids], z], -1).flatten(0, 1)
        mu = student(states)[0].view(len(z), len(ids), 12)
        action_loss = F.mse_loss(mu, batch['action'][burn:, ids])
        mask = batch['tick'][burn:, ids]
        latent_loss = (z-batch['terrain'][burn:, ids]).square().mean(-1)
        latent_loss = (latent_loss*mask).sum()/mask.sum().clamp(min=1)
        if 'weights' in batch:
            weight=batch['weights'][burn:,ids]
            error=(mu-batch['action'][burn:,ids]).square().mean(-1)
            action_loss=(error*weight).sum()/weight.sum()
            error=(z-batch['terrain'][burn:,ids]).square().mean(-1)
            latent_loss=(error*mask*weight).sum()/(mask*weight).sum().clamp(min=1)
        velocity_loss = F.mse_loss(2*student.velocity(states),
                                   2*batch['velocity'][burn:, ids].flatten(0, 1))
        loss = config.action_coef*action_loss+config.latent_coef*latent_loss+config.velocity_coef*velocity_loss
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite distillation loss')
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.)
        optimizer.step()
        totals += torch.stack([action_loss.detach(), latent_loss.detach(), velocity_loss.detach()])
        count += 1
    result = (totals/count).tolist()
    return dict(action_loss=result[0], latent_loss=result[1], velocity_loss=result[2])
