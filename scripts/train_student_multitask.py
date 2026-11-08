"""学生PPO联合采集路线和平地/自由速度，用同状态参考动作软约束保留步态。"""

import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import shutil
from pathlib import Path
import time

import torch

from domain_randomization.config import d435i_scene_config
from env.commands import free_training_config, nominal_flat_training_config
from env.perception.config import DepthConfig
from env.perception.depth_sensor import DepthSensor
from env.perception.runtime import VisualRuntime
from env.training_task import Go2Task
from env.gait import flat_terrain_mask
from rl.algorithms.perceptive_networks import PerceptiveActor, PrivilegedCritic
from rl.algorithms.ppo import PPO
from rl.config import TrainConfig
from rl.reference_targets import terrain_reference_weight
from rl.run_io import ROOT, append_csv, deployment_config, env_from_dict, load_policy, new_run, save_checkpoint, save_json


REFERENCE=ROOT/'outputs/20261106_135343_676539_student_ppo_seed92037/checkpoints/final.pt'
VISUAL=ROOT/'outputs/20261106_213606_439946_d435i_visual_adaptation_seed82200/checkpoints/final.pt'
FREE_REFERENCE=ROOT/'outputs/20261106_222656_882705_student_ppo_seed81110/checkpoints/final.pt'


def task_configs(source, strength, route_envs, free_envs, seed, flat_gait=False):
    route=d435i_scene_config(source)
    route.num_envs=route_envs;route.seed=seed;route.terrain.seed=seed+1000
    upper=min(50.,23.333333333333+26.666666666667*strength)
    route.camera.pitch_down_degrees=(20.+upper)/2
    route.randomization.camera_pitch_degrees=(20.,upper)
    route.command.mode='legacy';route.command.vx=(.3,.3);route.command.stop_script_fraction=0.
    route.command.stop_fraction=0.;route.episode_seconds=90.
    d=route.randomization;d.seed=seed+2000
    d.payload_max=.5*strength;d.mass_scale=(1-.05*strength,1+.05*strength)
    d.friction=(.8-.15*strength,.8+.2*strength);d.com_error=.003*strength
    d.gain_scale=(1-.05*strength,1+.05*strength);d.effort_scale=(1-.05*strength,1.)
    d.damping_scale=(1-.1*strength,1+.1*strength)
    for name,full in [('gyro_noise',.01),('gyro_bias',.005),('position_noise',.003),
                      ('velocity_noise',.05),('joint_bias',.005),('gravity_angle',.008726646),
                      ('depth_noise',.002),('depth_scale_error',.01),('depth_dropout',.01),
                      ('depth_patch_probability',.15),('frame_dropout',.01)]:
        setattr(d,name,full*strength)
    d.action_delay_steps=int(strength>=.5);d.frame_delay_steps=int(strength>=.5)
    d.enabled=strength>0
    # 视角由20～30逐步扩到20～50；四类侧墙始终启用，最终必须strength=1验收。
    free=free_training_config(route,True)
    free.num_envs=free_envs;free.seed=seed+1;free.randomization.seed=seed+2001
    free.randomization.payload_max=d.payload_max
    free.command.stop_script_fraction=.25;free.command.stop_hold_seconds=(8.,14.)
    free.training_reward.standing_velocity=40.
    free.training_reward.standing_joint_velocity=.1
    free.training_reward.standing_angular=5.
    free.training_reward.tracking_yaw_scale=.06
    free.training_reward.tracking_yaw=1.
    if strength==0:
        # 标称步态阶段：固定相机、物理和地形几何；自由任务也固定0.3直行。
        for cfg in (route,free):
            cfg.randomization.enabled=False
            cfg.camera.pitch_down_degrees=25.
            cfg.terrain.seed=56000
        # 标称阶段用审计同样的单布局/窄侧墙平地，避免视觉场景差异掩盖滑移。
        free=nominal_flat_training_config(route,free_envs,seed+1)
        free.command.vx=(.3,.3);free.command.wz=(0.,0.)
        free.command.stop_fraction=0.;free.command.anchor_fraction=0.
        free.command.stop_script_fraction=0.
    if flat_gait:
        # 这是标称步态课程的第一步，不替代后续复杂地形验收。
        route=deepcopy(free);route.num_envs=route_envs;route.seed=seed
    return route,free


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--resume')
    p.add_argument('--nominal-gait',action='store_true')
    p.add_argument('--preserve-reference-actions',action='store_true',help='直接沿用参考动作，跳过标称阶段髋目标改写')
    p.add_argument('--flat-gait-only',action='store_true')
    p.add_argument('--training-max-level',type=int,choices=(0,6,9,12,15,17),default=17)
    p.add_argument('--normal-foot-scale',type=float,default=0.)
    p.add_argument('--front-posture-scale',type=float,default=0.)
    p.add_argument('--reference',default=str(REFERENCE))
    p.add_argument('--terrain-reference',help='仅在标称复杂地形使用冻结专家标签；平地仍保留正常步态参考')
    p.add_argument('--terrain-reference-transition',type=float,default=0.,help='辅助标签在障碍前后平滑过渡的距离，0保持原切换方式')
    p.add_argument('--visual',default=str(VISUAL))
    p.add_argument('--free-reference',default=str(FREE_REFERENCE))
    p.add_argument('--free-anchor-weight',type=float,default=.5)
    p.add_argument('--iterations',type=int,default=100)
    p.add_argument('--route-envs',type=int,default=384)
    p.add_argument('--free-envs',type=int,default=128)
    p.add_argument('--horizon',type=int,default=128)
    p.add_argument('--strength',type=float,default=.25)
    p.add_argument('--anchor-coef',type=float,default=20.)
    p.add_argument('--actor-lr',type=float,default=5e-6)
    p.add_argument('--std',type=float,default=.04)
    p.add_argument('--seed',type=int,default=84000)
    p.add_argument('--route-std-scale',type=float,default=.5)
    p.add_argument('--free-std-scale',type=float,default=2.)
    p.add_argument('--critic-task-marker',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--freeze-proprio',action=argparse.BooleanOptionalAction,default=False)
    p.add_argument('--stop-script-fraction',type=float,default=.25)
    args=p.parse_args()
    if not (0<args.strength<=1 or (args.strength==0 and args.nominal_gait)):
        p.error('只有显式nominal-gait阶段允许strength=0，不能将其视作域随机化完成')
    if args.flat_gait_only and not (args.nominal_gait and args.strength==0):p.error('flat-gait-only仅用于关闭随机化的标称步态课程')
    if args.terrain_reference and not (args.nominal_gait and args.strength==0):
        p.error('terrain-reference仅用于关闭随机化的标称步态恢复')
    if args.terrain_reference_transition and not args.terrain_reference:
        p.error('terrain-reference-transition需要terrain-reference')
    torch.set_num_threads(1)
    if args.nominal_gait:args.free_reference=args.reference
    reference,source,reference_saved=load_policy(args.reference)
    reference.requires_grad_(False)
    free_reference,_,_=load_policy(args.free_reference)
    free_reference.requires_grad_(False)
    terrain_reference=None
    if args.terrain_reference:
        terrain_reference,terrain_source,terrain_saved=load_policy(args.terrain_reference)
        terrain_reference.requires_grad_(False)
    saved=torch.load(args.resume or args.reference,map_location='cuda:0',weights_only=False)
    training=TrainConfig(**saved['train_config'])
    training.horizon=args.horizon;training.batch_size=8192;training.epochs=5
    training.actor_lr=args.actor_lr;training.critic_lr=3e-4
    training.anchor_coef=args.anchor_coef;training.freeze_proprio=args.freeze_proprio
    training.target_action_std=args.std;training.adaptive_lr=False
    actor=PerceptiveActor('student',training);critic=PrivilegedCritic(training)
    actor.load_state_dict(saved['actor']);critic.load_state_dict(saved['critic'])
    if not args.resume:
        visual=torch.load(args.visual,map_location='cpu',weights_only=False)
        # 恢复195共享控制、本体/速度网络；只接入已适应D435i的视觉分支，不混淆来源。
        state={k.removeprefix('visual_encoder.'):v for k,v in visual['actor'].items() if k.startswith('visual_encoder.')}
        actor.visual_encoder.load_state_dict(state)
    share_normal_visual=(terrain_reference is not None and all(
        torch.equal(v.cpu(),reference.visual_encoder.state_dict()[k].cpu())
        for k,v in actor.visual_encoder.state_dict().items()))
    if terrain_reference is not None:
        # 编码器不同时各自维护GRU状态，冻结参考不能接收新学生的特征。
        for name in ('action_scale','kp','kd','physics_dt','decimation'):
            assert getattr(terrain_source,name)==getattr(source,name), f'专家控制配置不同：{name}'
    actor.visual_encoder.requires_grad_(False)
    if args.freeze_proprio:
        actor.history.requires_grad_(False);actor.velocity_head.requires_grad_(False)
    agent=PPO(actor=actor,critic=critic,config=training,device='cuda:0',
              actor_lr=training.actor_lr,critic_lr=training.critic_lr,gamma=training.gamma,
              lmbda=training.lmbda,epochs=training.epochs,epsilon=training.epsilon)
    # 蒸馏与PPO的优化器参数不同；跨阶段继承权重，只有PPO续训恢复优化器。
    if args.resume and saved['stage']=='student_ppo' and 'task_configs' in saved:
        agent.actor_optimizer.load_state_dict(saved['actor_optimizer'])
        agent.critic_optimizer.load_state_dict(saved['critic_optimizer'])
        for group in agent.actor_optimizer.param_groups:group['lr']=args.actor_lr
        for group in agent.critic_optimizer.param_groups:group['lr']=training.critic_lr
        agent.actor_optimizer.state.pop(agent.actor.log_std,None)
    with torch.no_grad():agent.actor.log_std.add_(torch.log(torch.tensor(args.std)/agent.actor.log_std.exp().mean()))
    route_cfg,free_cfg=task_configs(source,args.strength,args.route_envs,args.free_envs,args.seed,flat_gait=args.flat_gait_only)
    if args.terrain_reference_transition and args.terrain_reference_transition<=route_cfg.gait.flat_margin:
        p.error('参考过渡距离必须大于平地判定余量')
    if terrain_reference is not None:
        assert asdict(route_cfg.camera)==asdict(source.camera), '共享平地视觉特征需要相同的相机配置'
        assert reference_saved['env_config']['terrain'].get('surroundings'), '平地参考需要相同可见几何组'
    if not args.nominal_gait:free_cfg.command.stop_script_fraction=args.stop_script_fraction
    if args.front_posture_scale:
        from env.posture import configure_front_posture
        for cfg in (route_cfg,free_cfg):configure_front_posture(cfg,args.front_posture_scale)
    if args.nominal_gait or args.normal_foot_scale:
        from env.posture import configure_normal_gait
        scale=args.normal_foot_scale or args.front_posture_scale
        for cfg in (route_cfg,free_cfg):configure_normal_gait(cfg,scale)
    start=saved['iteration'];total=saved.get('total_steps',0)
    if saved['stage']=='distill' and 'source_ppo_iteration' in saved:
        # 视觉监督编号独立于PPO；交接元数据保留最近一次共享控制网络的PPO计数。
        start=saved['source_ppo_iteration'];total=saved['source_ppo_total_steps']
    folder=new_run('student_ppo',route_cfg,training)
    save_json(folder/'task_configs.json',dict(route=asdict(route_cfg),free=asdict(free_cfg),strength=args.strength))
    metadata=dict(status='running',parent=args.resume or args.reference,reference=args.reference,
                  reference_sha256=hashlib.sha256(Path(args.reference).read_bytes()).hexdigest(),
                  visual_source=args.visual if not args.resume else None,
                  initialization='195 common + D435i visual' if not args.resume else 'multitask continuation',
                  route_envs=args.route_envs,free_envs=args.free_envs,strength=args.strength,
                  arguments=vars(args),numerical_failures=0,
                  source_stage=saved['stage'],source_stage_iteration=saved['iteration'],ppo_start_iteration=start,
                  normal_gait_reward_revision='stance_speed_tail_v3',
                  terrain_contact_reward_revision=('low_support_deficit_v1' if route_cfg.gait.terrain_support else 'off'),
                  reference_target_revision=('geometry_transition_v2' if args.terrain_reference_transition else 'geometry_v2'),
                  terrain_reference=args.terrain_reference,
                  normal_reference_visual_shared=share_normal_visual,
                  terrain_reference_sha256=(hashlib.sha256(Path(args.terrain_reference).read_bytes()).hexdigest() if args.terrain_reference else None),
                  nominal_hip_prior=dict(enabled=args.nominal_gait and not args.preserve_reference_actions,neutral_target=args.flat_gait_only,limit_rad=route_cfg.training_reward.front_hip_deadband,
                    application='平地参考标签，执行动作不裁剪，约束服从PPO KL早停'))
    save_json(folder/'metadata.json',metadata);print(f'RUN_DIR={folder}',flush=True)
    tasks=[]
    try:
        route=Go2Task(route_cfg,curriculum=False);tasks.append(route)
        free=Go2Task(free_cfg,curriculum=False);tasks.append(free)
        terrain=route.env.scene.terrain
        # 保留已经学会的低台阶，逐级加入新高度；不改变地形或最终验收。
        active_levels=[level for level in (0,6,9,12,15,17) if level<=args.training_max_level]
        levels=torch.tensor(active_levels if route_cfg.terrain.obstacles else [0],device='cuda:0')
        highest_level=int(levels.max().item())
        terrain.terrain_levels[:]=levels[torch.arange(args.route_envs,device='cuda:0')%len(levels)]
        terrain.env_origins[:]=terrain.terrain_origins[terrain.terrain_levels,terrain.terrain_types]
        route.reset()
        visuals=[VisualRuntime(t,agent.actor.visual_encoder) for t in tasks]
        free_reference_visual=VisualRuntime(free,free_reference.visual_encoder)
        route_expert=terrain_reference if terrain_reference is not None else reference
        route_expert_saved=terrain_saved if terrain_reference is not None else reference_saved
        reference_visual=VisualRuntime(route,route_expert.visual_encoder,
            DepthSensor(route,DepthConfig(**route_expert_saved['env_config'].get('camera',{})),
                geometry_groups=[0,4] if route_expert_saved['env_config']['terrain'].get('surroundings') else [0],randomize=False))
        normal_reference_visual=None
        if terrain_reference is not None and not share_normal_visual:
            normal_reference_visual=VisualRuntime(route,reference.visual_encoder,
                DepthSensor(route,DepthConfig(**reference_saved['env_config'].get('camera',{})),
                    geometry_groups=[0,4],randomize=False))
        torch.manual_seed(args.seed)
        deployment=deployment_config(route);deployment['proprio_only']=False
        save_json(folder/'deployment.json',deployment)
        std_scale=torch.cat([torch.full((args.route_envs,1),args.route_std_scale,device='cuda:0'),
                             torch.full((args.free_envs,1),args.free_std_scale,device='cuda:0')])
        begin=time.monotonic()
        for iteration in range(start+1,start+args.iterations+1):
            sums=[dict(reward=0.,episodes=0,successes=0,falls=0,width_reward=0.,slip_reward=0.,touchdown_velocity_reward=0.,stance_speed_reward=0.,support_deficit_reward=0.,terrain_reference_fraction=0.,reference_transition_fraction=0.) for _ in tasks]
            highest_counts=dict(episodes=0,successes=0,falls=0)
            for _ in range(args.horizon):
                with torch.no_grad():
                    states=[];critics=[]
                    for t,v in zip(tasks,visuals):
                        s,c=t.inputs(v.latent)
                        if t is free and args.critic_task_marker:
                            # 自由任务没有路线进度；critic最后一维用-1标记任务，Actor仍不接收标签。
                            c[:,-1]=-1.
                        states.append(s);critics.append(c)
                    state=torch.cat(states);critic_state=torch.cat(critics)
                    action,sample=agent.sample_batch(state,critic_state,std_scale=std_scale)
                    ref_state,_=route.inputs(reference_visual.latent)
                    target=route_expert(ref_state)[0]
                    if terrain_reference is not None:
                        local_x=(route.robot.data.root_link_pos_w[:,0]-route.env.scene.env_origins[:,0]
                                 +route.config.terrain.spawn_x)
                        normal_latent=visuals[0].latent if normal_reference_visual is None else normal_reference_visual.latent
                        normal_state,_=route.inputs(normal_latent)
                        normal_target=reference(normal_state)[0]
                        # 仅切换辅助标签；执行动作始终来自当前学生，部署不依赖旧专家。
                        terrain_mask=~flat_terrain_mask(route,local_x)
                        weight=terrain_reference_weight(route,local_x,args.terrain_reference_transition)
                        if args.terrain_reference_transition:
                            sums[0]['reference_transition_fraction']+=(weight>terrain_mask.float()).float().mean().item()
                            target=torch.lerp(normal_target,target,weight[:,None])
                        else:
                            target=torch.where(terrain_mask[:,None],target,normal_target)
                        sums[0]['terrain_reference_fraction']+=weight.mean().item()
                    free_ref_state,_=free.inputs(free_reference_visual.latent)
                    free_target=free_reference(free_ref_state)[0]
                    if args.nominal_gait and not args.preserve_reference_actions:
                        # 只改变平地参考标签，不裁剪执行动作。其余关节保留参考轨迹，姿态项仍服从PPO的KL早停。
                        target=target.clone();free_target=free_target.clone()
                        hip_ids=[i for i,name in enumerate(route.robot.joint_names) if name.endswith('_hip_joint')]
                        local_x=(route.robot.data.root_link_pos_w[:,0]-route.env.scene.env_origins[:,0]
                                 +route.config.terrain.spawn_x)
                        straight=flat_terrain_mask(route,local_x)&(route.command()[:,2].abs()<.1)
                        limit=route_cfg.training_reward.front_hip_deadband/route_cfg.action_scale
                        for i in hip_ids:
                            target[:,i]=torch.where(straight,target[:,i].clamp(-limit,limit),target[:,i])
                            free_target[:,i]=free_target[:,i].clamp(-limit,limit)
                        if args.flat_gait_only:
                            # 平地先学接近默认姿态的髋目标，其余关节仍参考已有步态；不是执行动作限幅。
                            target[:,hip_ids]=0.;free_target[:,hip_ids]=0.
                    sample['anchor_mu']=torch.cat([target,free_target])
                    sample['anchor_weight']=torch.cat([torch.ones(args.route_envs,device='cuda:0'),
                        args.free_anchor_weight*(((free.command()[:,0]-.3).abs()>.01) | (free.command()[:,2].abs()>.03)).float()])
                    if args.nominal_gait:sample['anchor_weight'].fill_(1.)
                    rewards=[];dones=[];terms=[];next_critics=[];results=[]
                    for index,(t,v,a) in enumerate(zip(tasks,visuals,action.split([args.route_envs,args.free_envs]))):
                        reward,terminated,truncated,info=t.step(a)
                        v.update()
                        if index==0:
                            reference_visual.update()
                            if normal_reference_visual is not None:normal_reference_visual.update()
                        else:free_reference_visual.update()
                        _,next_critic=t.inputs()
                        if index==1 and args.critic_task_marker:next_critic[:,-1]=-1.
                        rewards.append(reward);dones.append(terminated|truncated);terms.append(terminated);next_critics.append(next_critic)
                        results.append(info)
                    agent.store_batch(sample,torch.cat(rewards),torch.cat(dones),torch.cat(terms),torch.cat(next_critics))
                    for index,(t,v,done,info) in enumerate(zip(tasks,visuals,dones,results)):
                        sums[index]['reward']+=info['terms']['tracking_xy'].mean().item()
                        if args.nominal_gait:
                            width_reward=info['terms'].get('normal_stance_width')
                            if width_reward is not None:sums[index]['width_reward']+=width_reward.mean().item()
                            sums[index]['slip_reward']+=info['terms']['slip'].mean().item()
                            touchdown_reward=info['terms'].get('normal_touchdown_velocity')
                            if touchdown_reward is not None:sums[index]['touchdown_velocity_reward']+=touchdown_reward.mean().item()
                            stance_speed_reward=info['terms'].get('normal_stance_speed')
                            if stance_speed_reward is not None:sums[index]['stance_speed_reward']+=stance_speed_reward.mean().item()
                            support_deficit=info['terms'].get('gait_terrain_deficit')
                            if support_deficit is not None:sums[index]['support_deficit_reward']+=support_deficit.mean().item()
                        ids=done.nonzero(as_tuple=False).flatten()
                        if len(ids):
                            sums[index]['episodes']+=len(ids)
                            sums[index]['successes']+=int(info['success'][ids].sum())
                            sums[index]['falls']+=int(info['fall'][ids].sum())
                            if index==0:
                                # reset前统计真实档位，避免把低台阶成功算作新高度的学习效果。
                                highest_ids=ids[terrain.terrain_levels[ids]==highest_level]
                                highest_counts['episodes']+=len(highest_ids)
                                highest_counts['successes']+=int(info['success'][highest_ids].sum())
                                highest_counts['falls']+=int(info['fall'][highest_ids].sum())
                            t.reset(ids)
                            v.reset(ids)
                            if index==0:
                                reference_visual.reset(ids)
                                if normal_reference_visual is not None:normal_reference_visual.reset(ids)
                            else:free_reference_visual.reset(ids)
            metrics=agent.update_batch()
            total+=args.horizon*(args.route_envs+args.free_envs)
            row=dict(iteration=iteration,total_steps=total,elapsed_seconds=time.monotonic()-begin,
                     route_tracking_reward=sums[0]['reward']/args.horizon,
                     free_tracking_reward=sums[1]['reward']/args.horizon,
                     route_episodes=sums[0]['episodes'],route_successes=sums[0]['successes'],
                     free_episodes=sums[1]['episodes'],free_successes=sums[1]['successes'],
                     route_max_level=highest_level,
                     route_max_level_episodes=highest_counts['episodes'],
                     route_max_level_successes=highest_counts['successes'],
                     route_max_level_falls=highest_counts['falls'],
                     numerical_failures=sum(t.numerical_failures for t in tasks),**metrics)
            if args.nominal_gait:
                for name,index in (('route',0),('free',1)):
                    row[name+'_width_reward']=sums[index]['width_reward']/args.horizon
                    row[name+'_slip_reward']=sums[index]['slip_reward']/args.horizon
                    row[name+'_touchdown_velocity_reward']=sums[index]['touchdown_velocity_reward']/args.horizon
                    row[name+'_stance_speed_reward']=sums[index]['stance_speed_reward']/args.horizon
                    row[name+'_support_deficit_reward']=sums[index]['support_deficit_reward']/args.horizon
            if terrain_reference is not None:
                row['terrain_reference_fraction']=sums[0]['terrain_reference_fraction']/args.horizon
                if args.terrain_reference_transition:
                    row['reference_transition_fraction']=sums[0]['reference_transition_fraction']/args.horizon
            append_csv(folder/'metrics.csv',row)
            with (folder/'train.log').open('a') as f:f.write(json.dumps(row)+'\n')
            if iteration==start+1 or (iteration-start)%10==0:print(json.dumps(row),flush=True)
            if (iteration-start)%25==0 or iteration==start+args.iterations:
                save_checkpoint(folder/'checkpoints/latest.pt',agent,iteration,route_cfg,training,'student_ppo',
                    task=route,deployment=deployment,total_steps=total,
                    task_configs=dict(route=asdict(route_cfg),free=asdict(free_cfg)),reference=args.reference,
                    exploration_std_scales=dict(route=args.route_std_scale,free=args.free_std_scale),
                    critic_free_task_marker=args.critic_task_marker,
                    preserve_reference_actions=args.preserve_reference_actions,
                    randomization_strength=args.strength,terrain_reference=args.terrain_reference,
                    terrain_reference_transition=args.terrain_reference_transition)
                if args.nominal_gait:
                    shutil.copyfile(folder/'checkpoints/latest.pt',folder/f'checkpoints/iteration_{iteration}.pt')
                if iteration==start+args.iterations:
                    shutil.copyfile(folder/'checkpoints/latest.pt',folder/'checkpoints/final.pt')
            metadata.update(iteration=iteration,numerical_failures=row['numerical_failures'])
            if (iteration-start)%10==0:save_json(folder/'metadata.json',metadata)
        metadata.update(status='finished',total_steps=total,
                        depth_capture_calls=sum(v.sensor.capture_count for v in visuals),
                        reference_capture_calls=reference_visual.sensor.capture_count,
                        free_reference_capture_calls=free_reference_visual.sensor.capture_count)
        save_json(folder/'metadata.json',metadata)
        print(f'MULTITASK_FINISHED={folder}',flush=True)
    except BaseException as error:
        metadata.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=str(error))
        save_json(folder/'metadata.json',metadata);raise
    finally:
        for task in tasks:task.close()


if __name__=='__main__':main()
