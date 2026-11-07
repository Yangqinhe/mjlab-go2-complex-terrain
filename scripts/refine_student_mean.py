"""均值动作闭环DAgger：冻结感知，仅监督修正动作MLP；不计作PPO更新。"""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import time

import torch
import torch.nn.functional as F

from env.gait import flat_terrain_mask
from env.perception.config import DepthConfig
from env.perception.depth_sensor import DepthSensor
from env.perception.runtime import VisualRuntime
from env.posture import configure_normal_gait
from env.training_task import Go2Task
from rl.algorithms.perceptive_networks import PrivilegedCritic
from rl.algorithms.ppo import PPO
from rl.config import TrainConfig
from rl.run_io import append_csv, deployment_config, load_policy, new_run, save_checkpoint, save_json
from scripts.train_student_multitask import task_configs


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--flat-reference',required=True)
    parser.add_argument('--terrain-reference',required=True)
    parser.add_argument('--route-envs',type=int,default=64)
    parser.add_argument('--free-envs',type=int,default=32)
    parser.add_argument('--max-level',type=int,choices=(0,6,9,12,15,17),default=6)
    parser.add_argument('--updates',type=int,default=200)
    parser.add_argument('--horizon',type=int,default=32)
    parser.add_argument('--epochs',type=int,default=2)
    parser.add_argument('--lr',type=float,default=3e-5)
    parser.add_argument('--seed',type=int,default=89950)
    args=parser.parse_args();torch.set_num_threads(1);torch.manual_seed(args.seed)
    student,source,saved=load_policy(args.checkpoint)
    normal,_,_=load_policy(args.flat_reference);normal.requires_grad_(False)
    terrain_expert,terrain_source,terrain_saved=load_policy(args.terrain_reference)
    terrain_expert.requires_grad_(False)
    for key,value in student.visual_encoder.state_dict().items():
        assert torch.equal(value,normal.visual_encoder.state_dict()[key]), '共享视觉特征需要同一冻结编码器'
    for name in ('action_scale','kp','kd','physics_dt','decimation'):
        assert getattr(source,name)==getattr(terrain_source,name), f'专家控制配置不一致：{name}'
    route_cfg,free_cfg=task_configs(source,0.,args.route_envs,args.free_envs,args.seed)
    assert asdict(route_cfg.camera)==asdict(source.camera)
    for cfg in (route_cfg,free_cfg):configure_normal_gait(cfg,1.)
    student.requires_grad_(False);student.actor.requires_grad_(True)
    training=TrainConfig(**saved['train_config']);training.actor_lr=args.lr
    critic=PrivilegedCritic(training).to('cuda:0');critic.load_state_dict(saved['critic']);critic.requires_grad_(False)
    agent=PPO(actor=student,critic=critic,config=training,device='cuda:0')
    agent.actor_optimizer=torch.optim.Adam(student.actor.parameters(),lr=args.lr)
    folder=new_run('mean_bc',route_cfg,training)
    metadata=dict(status='running',method='deterministic_student_state_dagger',reference_target_revision='geometry_v2',parent=args.checkpoint,
        flat_reference=args.flat_reference,terrain_reference=args.terrain_reference,
        terrain_reference_sha256=hashlib.sha256(Path(args.terrain_reference).read_bytes()).hexdigest(),
        ppo_iteration_unchanged=saved['iteration'],randomization_enabled=False,
        trainable='action MLP only; critic and log_std frozen',arguments=vars(args))
    save_json(folder/'metadata.json',metadata);print(f'MEAN_REFINEMENT_DIR={folder}',flush=True)
    tasks=[];began=time.monotonic();total=saved.get('total_steps',0)
    try:
        route=Go2Task(route_cfg,curriculum=False);tasks.append(route)
        free=Go2Task(free_cfg,curriculum=False);tasks.append(free)
        levels=torch.tensor([level for level in (0,6,9,12,15,17) if level<=args.max_level],device='cuda:0')
        terrain=route.env.scene.terrain
        terrain.terrain_levels[:]=levels[torch.arange(args.route_envs,device='cuda:0')%len(levels)]
        terrain.env_origins[:]=terrain.terrain_origins[terrain.terrain_levels,terrain.terrain_types];route.reset()
        visuals=[VisualRuntime(task,student.visual_encoder) for task in tasks]
        expert_visual=VisualRuntime(route,terrain_expert.visual_encoder,
            DepthSensor(route,DepthConfig(**terrain_saved['env_config'].get('camera',{})),geometry_groups=[0],randomize=False))
        hips=[i for i,name in enumerate(route.robot.joint_names) if name.endswith('_hip_joint')]
        deployment=deployment_config(route);save_json(folder/'deployment.json',deployment)
        for update in range(1,args.updates+1):
            states=[];targets=[];episodes=[0,0];successes=[0,0];rough_samples=0
            for _ in range(args.horizon):
                with torch.no_grad():
                    for index,(task,visual) in enumerate(zip(tasks,visuals)):
                        state,_=task.inputs(visual.latent);target=normal(state)[0].clone()
                        if index==0:
                            local_x=task.robot.data.root_link_pos_w[:,0]-task.env.scene.env_origins[:,0]+task.config.terrain.spawn_x
                            rough=~flat_terrain_mask(task,local_x)
                            expert_state,_=task.inputs(expert_visual.latent)
                            target=torch.where(rough[:,None],terrain_expert(expert_state)[0],target)
                            rough_samples+=int(rough.sum())
                        else:rough=torch.zeros(task.config.num_envs,dtype=torch.bool,device=task.device)
                        limit=task.config.training_reward.front_hip_deadband/task.config.action_scale
                        target[:,hips]=torch.where(rough[:,None],target[:,hips],target[:,hips].clamp(-limit,limit))
                        states.append(state.clone());targets.append(target)
                        # 采样与最终评估均使用学生均值，不注入教师动作或Gaussian噪声。
                        _,terminated,truncated,info=task.step(student(state)[0]);visual.update()
                        if index==0:expert_visual.update()
                        ids=(terminated|truncated).nonzero().flatten()
                        if len(ids):
                            episodes[index]+=len(ids);successes[index]+=int(info['success'][ids].sum())
                            task.reset(ids);visual.reset(ids)
                            if index==0:expert_visual.reset(ids)
            state=torch.cat(states);target=torch.cat(targets)
            for _ in range(args.epochs):
                loss=F.mse_loss(student(state)[0],target)
                if not torch.isfinite(loss):raise RuntimeError('均值动作监督损失非有限')
                agent.actor_optimizer.zero_grad();loss.backward()
                torch.nn.utils.clip_grad_norm_(student.actor.parameters(),.5);agent.actor_optimizer.step()
            failures=sum(task.numerical_failures for task in tasks)
            if failures:raise RuntimeError('均值闭环仿真出现数值异常，候选不采用')
            total+=(args.route_envs+args.free_envs)*args.horizon
            row=dict(update=update,elapsed_seconds=time.monotonic()-began,action_mse=loss.item(),
                route_episodes=episodes[0],route_successes=successes[0],free_episodes=episodes[1],
                free_successes=successes[1],rough_fraction=rough_samples/(args.horizon*args.route_envs),numerical_failures=failures)
            append_csv(folder/'metrics.csv',row)
            if update==1 or update%10==0:print(json.dumps(row),flush=True)
            if update%25==0 or update==args.updates:
                adaptation=dict(kind='deterministic_student_state_dagger',updates=update,
                    additional_env_steps=update*(args.route_envs+args.free_envs)*args.horizon,
                    ppo_iteration_unchanged=saved['iteration'])
                save_checkpoint(folder/'checkpoints/latest.pt',agent,saved['iteration'],route_cfg,training,
                    'mean_bc',task=route,deployment=deployment,total_steps=total,adaptation=adaptation)
                shutil.copyfile(folder/'checkpoints/latest.pt',folder/f'checkpoints/update_{update}.pt')
                if update==args.updates:shutil.copyfile(folder/'checkpoints/latest.pt',folder/'checkpoints/final.pt')
        metadata.update(status='finished',updates=args.updates,numerical_failures=0)
        save_json(folder/'metadata.json',metadata);print(f'MEAN_REFINEMENT_FINISHED={folder}',flush=True)
    except BaseException as error:
        metadata.update(status='failed',error=str(error));save_json(folder/'metadata.json',metadata);raise
    finally:
        for task in tasks:task.close()


if __name__=='__main__':main()
