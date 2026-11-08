"""统一教师监督视觉学生：路线和平地共同采样，只更新CNN/GRU，保留已学步态。"""

import argparse
from dataclasses import asdict
import hashlib
from pathlib import Path
import time

import torch

from env.commands import nominal_flat_training_config
from env.perception.runtime import VisualRuntime
from env.training_task import Go2Task
from rl.algorithms.perceptive_networks import PerceptiveActor, PrivilegedCritic
from rl.algorithms.ppo import PPO
from rl.config import DistillConfig, TrainConfig
from rl.distillation import collect_sequences, merge_sequence_batches, update_sequences
from rl.run_io import append_csv, deployment_config, load_policy, new_run, save_checkpoint, save_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--teacher',required=True)
    parser.add_argument('--resume',help='恢复本脚本或复用已有学生视觉；共享网络始终来自教师')
    parser.add_argument('--visual-init',help='换教师时仅复用已有学生视觉，重建优化器和本阶段计数')
    parser.add_argument('--num-envs',type=int,default=64)
    parser.add_argument('--free-envs',type=int,default=64)
    parser.add_argument('--iterations',type=int,default=100)
    parser.add_argument('--visual-lr',type=float,default=1e-4)
    parser.add_argument('--teacher-mix',type=float,default=1.)
    parser.add_argument('--visual-steps',type=int,default=20)
    parser.add_argument('--burn-in',type=int,default=5)
    parser.add_argument('--sequence-batch',type=int,default=16)
    parser.add_argument('--training-levels',type=int,nargs='+',default=[0,6,9,12,15,17])
    parser.add_argument('--seed',type=int,default=90020)
    args=parser.parse_args()
    if args.resume and args.visual_init:parser.error('resume与visual-init分别用于续训和换教师初始化，不能同时指定')
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    teacher,config,teacher_saved=load_policy(args.teacher)
    assert teacher.mode=='teacher'
    teacher.requires_grad_(False)
    # SHA约束恢复时的监督目标，避免同一视觉优化器悄悄换教师。
    teacher_sha=hashlib.sha256(Path(args.teacher).read_bytes()).hexdigest()
    saved=torch.load(args.resume,map_location='cuda:0',weights_only=False) if args.resume else None
    training=TrainConfig(**teacher_saved['train_config'])
    student=PerceptiveActor('student',training).to('cuda:0')
    if saved:student.load_state_dict(saved['actor'])
    if args.visual_init:
        visual_saved=torch.load(args.visual_init,map_location='cuda:0',weights_only=False)
        student.visual_encoder.load_state_dict({k.removeprefix('visual_encoder.'):v
            for k,v in visual_saved['actor'].items() if k.startswith('visual_encoder.')})
    student.copy_common(teacher)
    student.requires_grad_(False)
    student.visual_encoder.requires_grad_(True)
    critic=PrivilegedCritic(training).to('cuda:0')
    critic.load_state_dict(teacher_saved['critic']);critic.requires_grad_(False)
    agent=PPO(actor=student,critic=critic,config=training,device='cuda:0')
    agent.actor_optimizer=torch.optim.Adam(student.visual_encoder.parameters(),lr=args.visual_lr)
    start=0
    if saved and saved.get('algorithm')=='gait_student_distillation':
        assert saved['teacher_sha256']==teacher_sha
        agent.actor_optimizer.load_state_dict(saved['actor_optimizer'])
        start=saved['iteration']
    for group in agent.actor_optimizer.param_groups:group['lr']=args.visual_lr
    distill=DistillConfig(num_envs=args.num_envs,visual_steps=args.visual_steps,burn_in=args.burn_in,
        sequence_batch=args.sequence_batch,visual_lr=args.visual_lr,common_lr=0.,freeze_common=True,
        action_coef=1.,latent_coef=.1,velocity_coef=0.,teacher_mix=args.teacher_mix,
        training_levels=tuple(args.training_levels))
    config.num_envs,config.seed=args.num_envs,args.seed
    config.randomization.enabled=False
    config.terrain.seed=56000
    free_config=nominal_flat_training_config(config,args.free_envs,args.seed+1)
    folder=new_run('gait_student_distillation',config,training)
    metadata=dict(status='running',teacher=args.teacher,teacher_sha256=teacher_sha,parent=args.resume,
        visual_initialization=args.visual_init,
        arguments=vars(args),teacher_privileged=True,student_privileged=False,
        student_goal_complete=False,randomization_enabled=False)
    save_json(folder/'metadata.json',metadata)
    save_json(folder/'distillation.json',asdict(distill))
    save_json(folder/'task_configs.json',dict(route=asdict(config),free=asdict(free_config)))
    print('RUN_DIR='+str(folder),flush=True)
    tasks=[]
    try:
        route=Go2Task(config,curriculum=False);tasks.append(route)
        free=Go2Task(free_config,curriculum=False);tasks.append(free)
        terrain=route.env.scene.terrain
        levels=torch.tensor(args.training_levels,device='cuda:0')
        terrain.terrain_levels[:]=levels[torch.arange(args.num_envs,device='cuda:0')%len(levels)]
        terrain.env_origins[:]=terrain.terrain_origins[terrain.terrain_levels,terrain.terrain_types]
        route.reset()
        visuals=[VisualRuntime(task,student.visual_encoder) for task in tasks]
        deployment=deployment_config(route)
        began=time.monotonic()
        for iteration in range(start+1,start+args.iterations+1):
            batches=[];sampling={}
            for name,task,visual in zip(('route','free'),tasks,visuals):
                batch,stats=collect_sequences(task,visual,teacher,student,
                    (args.visual_steps+args.burn_in)*5,args.teacher_mix)
                batches.append(batch)
                sampling.update({name+'_'+key:value for key,value in stats.items()})
            if any(task.numerical_failures for task in tasks):raise RuntimeError('学生采样数值异常，本批保留拒用')
            metrics=update_sequences(student,agent.actor_optimizer,merge_sequence_batches(batches),distill)
            row=dict(iteration=iteration,elapsed_seconds=time.monotonic()-began,numerical_failures=0,
                     **sampling,**metrics)
            append_csv(folder/'metrics.csv',row)
            print(row,flush=True)
            if iteration%25==0 or iteration==start+args.iterations:
                extra=dict(task=route,deployment=deployment,algorithm='gait_student_distillation',
                    teacher=args.teacher,teacher_sha256=teacher_sha,
                    visual_initialization=args.visual_init,
                    source_teacher_iteration=teacher_saved['iteration'],
                    source_ppo_iteration=0,source_ppo_total_steps=0,source_visual_iteration=iteration,
                    privileged_input=False,student_acceptance=False,
                    task_configs=dict(route=asdict(config),free=asdict(free_config)))
                save_checkpoint(folder/f'checkpoints/iteration_{iteration}.pt',agent,iteration,config,training,'distill',**extra)
                save_checkpoint(folder/'checkpoints/latest.pt',agent,iteration,config,training,'distill',**extra)
                if iteration==start+args.iterations:
                    save_checkpoint(folder/'checkpoints/final.pt',agent,iteration,config,training,'distill',**extra)
        metadata.update(status='finished',iteration=iteration,numerical_failures=0)
        save_json(folder/'metadata.json',metadata)
        print('GAIT_STUDENT_DISTILLATION_FINISHED='+str(folder),flush=True)
    except BaseException as error:
        metadata.update(status='failed',error=str(error));save_json(folder/'metadata.json',metadata);raise
    finally:
        for task in tasks:task.close()


if __name__=='__main__':main()
