"""同一学生PPO联合采集复杂路线和自由速度，用195同状态动作约束保留步态。"""

import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import torch

from domain_randomization.config import d435i_scene_config
from env.commands import free_training_config
from env.perception.config import DepthConfig
from env.perception.depth_sensor import DepthSensor
from env.perception.runtime import VisualRuntime
from env.training_task import Go2Task
from env.gait import flat_mask
from rl.algorithms.perceptive_networks import PerceptiveActor, PrivilegedCritic
from rl.algorithms.ppo import PPO
from rl.config import TrainConfig
from rl.run_io import ROOT, append_csv, deployment_config, env_from_dict, load_policy, new_run, save_checkpoint, save_json


REFERENCE=ROOT/'outputs/20261106_135343_676539_student_ppo_seed92037/checkpoints/final.pt'
VISUAL=ROOT/'outputs/20261106_213606_439946_d435i_visual_adaptation_seed82200/checkpoints/final.pt'
FREE_REFERENCE=ROOT/'outputs/20261106_222656_882705_student_ppo_seed81110/checkpoints/final.pt'


def task_configs(source, strength, route_envs, free_envs, seed):
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
        free.command.vx=(.3,.3);free.command.wz=(0.,0.)
        free.command.stop_fraction=0.;free.command.anchor_fraction=0.
        free.command.stop_script_fraction=0.
    return route,free


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--resume')
    p.add_argument('--nominal-gait',action='store_true')
    p.add_argument('--normal-foot-scale',type=float,default=0.)
    p.add_argument('--front-posture-scale',type=float,default=0.)
    p.add_argument('--reference',default=str(REFERENCE))
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
    torch.set_num_threads(1)
    if args.nominal_gait:args.free_reference=args.reference
    reference,source,reference_saved=load_policy(args.reference)
    reference.requires_grad_(False)
    free_reference,_,_=load_policy(args.free_reference)
    free_reference.requires_grad_(False)
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
    actor.visual_encoder.requires_grad_(False)
    if args.freeze_proprio:
        actor.history.requires_grad_(False);actor.velocity_head.requires_grad_(False)
    agent=PPO(actor=actor,critic=critic,config=training,device='cuda:0',
              actor_lr=training.actor_lr,critic_lr=training.critic_lr,gamma=training.gamma,
              lmbda=training.lmbda,epochs=training.epochs,epsilon=training.epsilon)
    if args.resume and 'task_configs' in saved:
        agent.actor_optimizer.load_state_dict(saved['actor_optimizer'])
        agent.critic_optimizer.load_state_dict(saved['critic_optimizer'])
        for group in agent.actor_optimizer.param_groups:group['lr']=args.actor_lr
        for group in agent.critic_optimizer.param_groups:group['lr']=training.critic_lr
        agent.actor_optimizer.state.pop(agent.actor.log_std,None)
    with torch.no_grad():agent.actor.log_std.add_(torch.log(torch.tensor(args.std)/agent.actor.log_std.exp().mean()))
    route_cfg,free_cfg=task_configs(source,args.strength,args.route_envs,args.free_envs,args.seed)
    if not args.nominal_gait:free_cfg.command.stop_script_fraction=args.stop_script_fraction
    if args.front_posture_scale:
        from env.posture import configure_front_posture
        for cfg in (route_cfg,free_cfg):configure_front_posture(cfg,args.front_posture_scale)
    if args.nominal_gait or args.normal_foot_scale:
        from env.posture import configure_normal_gait
        scale=args.normal_foot_scale or args.front_posture_scale
        for cfg in (route_cfg,free_cfg):configure_normal_gait(cfg,scale)
    folder=new_run('student_ppo',route_cfg,training)
    save_json(folder/'task_configs.json',dict(route=asdict(route_cfg),free=asdict(free_cfg),strength=args.strength))
    metadata=dict(status='running',parent=args.resume or args.reference,reference=args.reference,
                  reference_sha256=hashlib.sha256(Path(args.reference).read_bytes()).hexdigest(),
                  visual_source=args.visual if not args.resume else None,
                  initialization='195 common + D435i visual' if not args.resume else 'multitask continuation',
                  route_envs=args.route_envs,free_envs=args.free_envs,strength=args.strength,
                  arguments=vars(args),numerical_failures=0,
                  nominal_hip_prior=dict(enabled=args.nominal_gait,limit_rad=route_cfg.training_reward.front_hip_deadband,
                    application='平地参考标签，执行动作不裁剪，约束服从PPO KL早停'))
    save_json(folder/'metadata.json',metadata);print(f'RUN_DIR={folder}',flush=True)
    tasks=[]
    try:
        route=Go2Task(route_cfg,curriculum=False);tasks.append(route)
        free=Go2Task(free_cfg,curriculum=False);tasks.append(free)
        terrain=route.env.scene.terrain
        levels=torch.tensor([0,6,9,12,15,17],device='cuda:0')
        terrain.terrain_levels[:]=levels[torch.arange(args.route_envs,device='cuda:0')%len(levels)]
        terrain.env_origins[:]=terrain.terrain_origins[terrain.terrain_levels,terrain.terrain_types]
        route.reset()
        visuals=[VisualRuntime(t,agent.actor.visual_encoder) for t in tasks]
        free_reference_visual=VisualRuntime(free,free_reference.visual_encoder)
        reference_visual=VisualRuntime(route,reference.visual_encoder,
            DepthSensor(route,DepthConfig(**reference_saved['env_config'].get('camera',{})),
                geometry_groups=[0,4] if reference_saved['env_config']['terrain'].get('surroundings') else [0],randomize=False))
        torch.manual_seed(args.seed)
        deployment=deployment_config(route);deployment['proprio_only']=False
        save_json(folder/'deployment.json',deployment)
        start=saved['iteration'];total=saved.get('total_steps',0)
        std_scale=torch.cat([torch.full((args.route_envs,1),args.route_std_scale,device='cuda:0'),
                             torch.full((args.free_envs,1),args.free_std_scale,device='cuda:0')])
        begin=time.monotonic()
        for iteration in range(start+1,start+args.iterations+1):
            sums=[dict(reward=0.,episodes=0,successes=0,falls=0) for _ in tasks]
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
                    target=reference(ref_state)[0]
                    free_ref_state,_=free.inputs(free_reference_visual.latent)
                    free_target=free_reference(free_ref_state)[0]
                    if args.nominal_gait:
                        # 只改变平地参考标签，不裁剪执行动作。其余关节保留参考轨迹，姿态项仍服从PPO的KL早停。
                        target=target.clone();free_target=free_target.clone()
                        hip_ids=[i for i,name in enumerate(route.robot.joint_names) if name.endswith('_hip_joint')]
                        local_x=(route.robot.data.root_link_pos_w[:,0]-route.env.scene.env_origins[:,0]
                                 +route.config.terrain.spawn_x)
                        straight=flat_mask(route,local_x)&(route.command()[:,2].abs()<.1)
                        limit=route_cfg.training_reward.front_hip_deadband/route_cfg.action_scale
                        for i in hip_ids:
                            target[:,i]=torch.where(straight,target[:,i].clamp(-limit,limit),target[:,i])
                            free_target[:,i]=free_target[:,i].clamp(-limit,limit)
                    sample['anchor_mu']=torch.cat([target,free_target])
                    sample['anchor_weight']=torch.cat([torch.ones(args.route_envs,device='cuda:0'),
                        args.free_anchor_weight*(((free.command()[:,0]-.3).abs()>.01) | (free.command()[:,2].abs()>.03)).float()])
                    if args.nominal_gait:sample['anchor_weight'].fill_(1.)
                    rewards=[];dones=[];terms=[];next_critics=[];results=[]
                    for index,(t,v,a) in enumerate(zip(tasks,visuals,action.split([args.route_envs,args.free_envs]))):
                        reward,terminated,truncated,info=t.step(a)
                        v.update()
                        if index==0:reference_visual.update()
                        else:free_reference_visual.update()
                        _,next_critic=t.inputs()
                        if index==1 and args.critic_task_marker:next_critic[:,-1]=-1.
                        rewards.append(reward);dones.append(terminated|truncated);terms.append(terminated);next_critics.append(next_critic)
                        results.append(info)
                    agent.store_batch(sample,torch.cat(rewards),torch.cat(dones),torch.cat(terms),torch.cat(next_critics))
                    for index,(t,v,done,info) in enumerate(zip(tasks,visuals,dones,results)):
                        sums[index]['reward']+=info['terms']['tracking_xy'].mean().item()
                        ids=done.nonzero(as_tuple=False).flatten()
                        if len(ids):
                            sums[index]['episodes']+=len(ids)
                            sums[index]['successes']+=int(info['success'][ids].sum())
                            sums[index]['falls']+=int(info['fall'][ids].sum())
                            t.reset(ids)
                            v.reset(ids)
                            if index==0:reference_visual.reset(ids)
                            else:free_reference_visual.reset(ids)
            metrics=agent.update_batch()
            total+=args.horizon*(args.route_envs+args.free_envs)
            row=dict(iteration=iteration,total_steps=total,elapsed_seconds=time.monotonic()-begin,
                     route_tracking_reward=sums[0]['reward']/args.horizon,
                     free_tracking_reward=sums[1]['reward']/args.horizon,
                     route_episodes=sums[0]['episodes'],route_successes=sums[0]['successes'],
                     free_episodes=sums[1]['episodes'],free_successes=sums[1]['successes'],
                     numerical_failures=sum(t.numerical_failures for t in tasks),**metrics)
            append_csv(folder/'metrics.csv',row)
            with (folder/'train.log').open('a') as f:f.write(json.dumps(row)+'\n')
            if iteration==start+1 or (iteration-start)%10==0:print(json.dumps(row),flush=True)
            if (iteration-start)%25==0 or iteration==start+args.iterations:
                save_checkpoint(folder/'checkpoints/latest.pt',agent,iteration,route_cfg,training,'student_ppo',
                    task=route,deployment=deployment,total_steps=total,
                    task_configs=dict(route=asdict(route_cfg),free=asdict(free_cfg)),reference=args.reference,
                    exploration_std_scales=dict(route=args.route_std_scale,free=args.free_std_scale),
                    critic_free_task_marker=args.critic_task_marker,
                    randomization_strength=args.strength)
                if iteration==start+args.iterations:
                    import shutil
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
