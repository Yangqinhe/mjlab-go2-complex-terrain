"""用同状态旧相机参考进行顺序视觉适应，训练CNN/GRU而保留已学习的控制网络。"""

import argparse
import copy
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from domain_randomization.config import d435i_scene_config
from env.perception.config import DepthConfig
from env.perception.depth_sensor import DepthSensor
from env.training_task import Go2Task
from rl.run_io import ROOT, append_csv, deployment_config, load_policy, save_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--num-envs',type=int,default=128)
    parser.add_argument('--iterations',type=int,default=250)
    parser.add_argument('--visual-steps',type=int,default=10)
    parser.add_argument('--burn-in',type=int,default=2)
    parser.add_argument('--seed',type=int,default=82200)
    parser.add_argument('--visual-init',help='仅初始化学生视觉，参考仍使用原始来源视觉')
    parser.add_argument('--latent-coef',type=float,default=.1)
    args=parser.parse_args()
    if args.visual_steps<=args.burn_in:parser.error('visual-steps必须大于burn-in')
    torch.set_num_threads(1);torch.manual_seed(args.seed)
    student,original,saved=load_policy(args.checkpoint)
    reference=copy.deepcopy(student).requires_grad_(False)
    if args.visual_init:
        phi=torch.load(args.visual_init,map_location='cpu',weights_only=False)
        student.visual_encoder.load_state_dict({k.removeprefix('visual_encoder.'):v
            for k,v in phi['actor'].items() if k.startswith('visual_encoder.')})
    student.requires_grad_(False);student.visual_encoder.requires_grad_(True)
    optimizer=torch.optim.Adam(student.visual_encoder.parameters(),lr=2e-5)
    config=d435i_scene_config(original);config.num_envs=args.num_envs
    config.seed=args.seed;config.randomization.seed=args.seed;config.terrain.seed=args.seed+1000
    # 顺序监督先对齐采集时刻；帧丢失和传输延迟在后续PPO/评估恢复。
    config.randomization.frame_delay_steps=0;config.randomization.frame_dropout=0.
    folder=ROOT/'outputs'/f'{datetime.now():%Y%m%d_%H%M%S_%f}_d435i_visual_adaptation_seed{args.seed}'
    (folder/'checkpoints').mkdir(parents=True)
    metadata=dict(status='running',parent=str(Path(args.checkpoint).resolve()),
                  source_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
                  purpose='20-50deg D435i nominal FOV and wall-scene visual adaptation',
                  ppo_iteration_unchanged=saved['iteration'],updates=0,additional_env_steps=0,
                  visual_init=args.visual_init,latent_coef=args.latent_coef,
                  trainable='visual_encoder only',reference='same state, old25deg camera, terrain geometry group0')
    save_json(folder/'config.json',dict(env=asdict(config),arguments=vars(args)))
    save_json(folder/'metadata.json',metadata)
    print(f'VISUAL_ADAPTATION_DIR={folder}',flush=True)
    task=Go2Task(config,curriculum=False)
    terrain=task.env.scene.terrain
    levels=torch.tensor([0,6,9,12,15,17],device=task.device)
    terrain.terrain_levels[:]=levels[torch.arange(args.num_envs,device=task.device)%len(levels)]
    terrain.env_origins[:]=terrain.terrain_origins[terrain.terrain_levels,terrain.terrain_types]
    task.reset()
    sensor=DepthSensor(task)
    clean=DepthSensor(task,DepthConfig(),geometry_groups=[0],randomize=False)
    n,device=args.num_envs,task.device
    pair=torch.zeros(n,2,48,64,device=device,dtype=torch.float16)
    clean_pair=torch.zeros_like(pair)
    hidden=torch.zeros(n,128,device=device);reference_hidden=torch.zeros_like(hidden)
    latent=torch.zeros(n,64,device=device);reference_latent=torch.zeros_like(latent)
    ids_all=torch.arange(n,device=device)

    def reset_images(ids,hidden,reference_hidden,latent,reference_latent):
        sensor.reset_mount(ids)
        a,b=sensor.capture(),clean.capture()
        pair[ids,0]=a[ids];pair[ids,1]=a[ids]
        clean_pair[ids,0]=b[ids];clean_pair[ids,1]=b[ids]
        zeros=torch.zeros(len(ids),128,device=device)
        # 同时记录重置时的第一帧编码，避免丢失额外reset帧造成RNN时序错配。
        z,h=student.visual_encoder(pair[ids],task.proprio()[ids],zeros)
        with torch.no_grad():rz,rh=reference.visual_encoder(clean_pair[ids],task.proprio()[ids],zeros)
        return (hidden.index_copy(0,ids,h),reference_hidden.index_copy(0,ids,rh),
                latent.index_copy(0,ids,z),reference_latent.index_copy(0,ids,rz))

    def checkpoint(iteration):
        payload=dict(saved)
        payload.update(actor=student.state_dict(),env_config=asdict(config),
                       deployment={**deployment_config(task),"proprio_only":False},
                       adaptation=dict(source=str(Path(args.checkpoint).resolve()),updates=iteration,
                                       env_steps=iteration*args.visual_steps*5*n,kind='camera_scene_consistency'),
                       visual_optimizer=optimizer.state_dict())
        # iteration和total_steps保留原PPO计数；视觉优化次数在adaptation单独记录。
        temporary=folder/'checkpoints/latest.tmp'
        torch.save(payload,temporary);temporary.replace(folder/'checkpoints/latest.pt')
        if iteration==args.iterations:torch.save(payload,folder/'checkpoints/final.pt')

    begin=time.monotonic()
    try:
        hidden,reference_hidden,latent,reference_latent=reset_images(ids_all,hidden,reference_hidden,latent,reference_latent)
        for iteration in range(1,args.iterations+1):
            hidden=hidden.detach();reference_hidden=reference_hidden.detach()
            latent=latent.detach();reference_latent=reference_latent.detach()
            action_losses=[];latent_losses=[];episodes=0
            for frame in range(args.visual_steps):
                for _ in range(5):
                    with torch.no_grad():
                        state,_=task.inputs(reference_latent)
                        action=reference(state)[0]
                        _,terminated,truncated,info=task.step(action)
                    ids=(terminated|truncated).nonzero(as_tuple=False).flatten()
                    if len(ids):
                        episodes+=len(ids);task.reset(ids)
                        hidden,reference_hidden,latent,reference_latent=reset_images(ids,hidden,reference_hidden,latent,reference_latent)
                a,b=sensor.capture(),clean.capture()
                pair[:,0]=pair[:,1];pair[:,1]=a
                clean_pair[:,0]=clean_pair[:,1];clean_pair[:,1]=b
                # clone防止下一帧原地更新双帧缓存破坏卷积反向所需输入。
                latent,hidden=student.visual_encoder(pair.clone(),task.proprio().clone(),hidden)
                with torch.no_grad():
                    reference_latent,reference_hidden=reference.visual_encoder(clean_pair,task.proprio(),reference_hidden)
                    reference_states,_=task.inputs(reference_latent)
                    target=reference(reference_states)[0]
                if frame>=args.burn_in:
                    states,_=task.inputs(latent)
                    action_losses.append(F.mse_loss(student(states)[0],target))
                    latent_losses.append(F.mse_loss(latent,reference_latent))
            action_loss=torch.stack(action_losses).mean();latent_loss=torch.stack(latent_losses).mean()
            loss=action_loss+args.latent_coef*latent_loss
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite visual adaptation loss')
            optimizer.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(student.visual_encoder.parameters(),1.);optimizer.step()
            row=dict(iteration=iteration,action_loss=action_loss.item(),latent_loss=latent_loss.item(),
                     episodes=episodes,numerical_failures=task.numerical_failures,
                     elapsed_seconds=time.monotonic()-begin)
            append_csv(folder/'metrics.csv',row)
            if iteration==1 or iteration%10==0:print(json.dumps(row),flush=True)
            if iteration%25==0 or iteration==args.iterations:checkpoint(iteration)
            metadata.update(updates=iteration,additional_env_steps=iteration*args.visual_steps*5*n,
                            numerical_failures=task.numerical_failures)
            if iteration%10==0:save_json(folder/'metadata.json',metadata)
        changed=[k for k in saved['actor'] if not torch.equal(saved['actor'][k].cpu(),student.state_dict()[k].cpu())]
        assert changed and all(k.startswith('visual_encoder.') for k in changed)
        metadata.update(status='finished_needs_closed_loop_evaluation',changed_parameter_keys=changed,
                        depth_calls=sensor.capture_count,reference_depth_calls=clean.capture_count)
        save_json(folder/'metadata.json',metadata)
        print(f'VISUAL_ADAPTATION_FINISHED={folder}',flush=True)
    except BaseException as error:
        metadata.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=str(error))
        save_json(folder/'metadata.json',metadata);raise
    finally:task.close()


if __name__=='__main__':main()
