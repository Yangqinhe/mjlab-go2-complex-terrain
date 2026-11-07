"""标称平地监督姿态适应：只更新动作MLP，髋参考为0，其余关节参考原策略；不是PPO更新。"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from env.perception.runtime import VisualRuntime
from env.training_task import Go2Task
from rl.algorithms.perceptive_networks import PrivilegedCritic
from rl.algorithms.ppo import PPO
from rl.config import TrainConfig
from rl.run_io import (load_policy, new_run, save_checkpoint, deployment_config,
                       save_json, append_csv)
from scripts.train_student_multitask import task_configs


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--reference',required=True)
    parser.add_argument('--num-envs',type=int,default=128)
    parser.add_argument('--updates',type=int,default=60)
    parser.add_argument('--horizon',type=int,default=16)
    parser.add_argument('--epochs',type=int,default=2)
    parser.add_argument('--lr',type=float,default=1e-4)
    parser.add_argument('--seed',type=int,default=89500)
    args=parser.parse_args();torch.set_num_threads(1);torch.manual_seed(args.seed)
    student,source,saved=load_policy(args.checkpoint)
    reference,_,_=load_policy(args.reference);reference.requires_grad_(False)
    # 两个来源使用同一冻结视觉分支才能共享latent；不允许把不同编码器的特征混用。
    for name,value in student.visual_encoder.state_dict().items():
        if not torch.equal(value,reference.visual_encoder.state_dict()[name]):
            raise ValueError('参考视觉编码器不同，不能共享视觉latent')
    config,_=task_configs(source,0.,args.num_envs,1,args.seed,flat_gait=True)
    student.requires_grad_(False);student.actor.requires_grad_(True)
    training=TrainConfig(**saved['train_config']);training.actor_lr=args.lr
    critic=PrivilegedCritic(training).to('cuda:0');critic.load_state_dict(saved['critic']);critic.requires_grad_(False)
    agent=PPO(actor=student,critic=critic,config=training,device='cuda:0')
    agent.actor_optimizer=torch.optim.Adam(student.actor.parameters(),lr=args.lr)
    folder=new_run('posture_bc',config,training)
    metadata=dict(status='running',parent=args.checkpoint,reference=args.reference,
        method='supervised_posture_adaptation',ppo_iteration_unchanged=saved['iteration'],
        trainable='action MLP only',randomization_enabled=False,arguments=vars(args))
    save_json(folder/'metadata.json',metadata);print(f'POSTURE_ADAPTATION_DIR={folder}',flush=True)
    task=Go2Task(config,curriculum=False)
    hips=[i for i,name in enumerate(task.robot.joint_names) if name.endswith('_hip_joint')]
    deployment=deployment_config(task);save_json(folder/'deployment.json',deployment)
    began=time.monotonic();total=saved.get('total_steps',0)
    try:
        visual=VisualRuntime(task,student.visual_encoder)
        for update in range(1,args.updates+1):
            states=[];targets=[];tracking=0.;episodes=0
            for _ in range(args.horizon):
                with torch.no_grad():
                    state,_=task.inputs(visual.latent)
                    target=reference(state)[0].clone();target[:,hips]=0.
                    states.append(state.clone());targets.append(target)
                    _,terminated,truncated,info=task.step(student(state)[0]);visual.update()
                    tracking+=info['terms']['tracking_xy'].mean().item()
                    ids=(terminated|truncated).nonzero().flatten()
                    if len(ids):task.reset(ids);visual.reset(ids);episodes+=len(ids)
            state=torch.cat(states);target=torch.cat(targets)
            for _ in range(args.epochs):
                mu=student(state)[0];loss=F.mse_loss(mu,target)
                if not torch.isfinite(loss):raise RuntimeError('监督姿态损失非有限')
                agent.actor_optimizer.zero_grad();loss.backward()
                torch.nn.utils.clip_grad_norm_(student.actor.parameters(),.5);agent.actor_optimizer.step()
            total+=args.num_envs*args.horizon
            row=dict(update=update,elapsed_seconds=time.monotonic()-began,action_mse=loss.item(),
                     tracking_reward=tracking/args.horizon,episodes=episodes,numerical_failures=task.numerical_failures)
            append_csv(folder/'metrics.csv',row)
            if update==1 or update%10==0:print(json.dumps(row),flush=True)
            if update%25==0 or update==args.updates:
                adaptation=dict(kind='supervised_posture',updates=update,
                                additional_env_steps=update*args.num_envs*args.horizon,
                                ppo_iteration_unchanged=saved['iteration'])
                save_checkpoint(folder/'checkpoints/latest.pt',agent,saved['iteration'],config,training,
                    'posture_bc',task=task,deployment=deployment,total_steps=total,adaptation=adaptation)
                if update==args.updates:
                    import shutil
                    shutil.copyfile(folder/'checkpoints/latest.pt',folder/'checkpoints/final.pt')
        metadata.update(status='finished',updates=args.updates,numerical_failures=task.numerical_failures)
        save_json(folder/'metadata.json',metadata);print(f'POSTURE_ADAPTATION_FINISHED={folder}',flush=True)
    except BaseException as error:
        metadata.update(status='failed',error=str(error));save_json(folder/'metadata.json',metadata);raise
    finally:task.close()


if __name__=='__main__':main()
