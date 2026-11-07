"""视觉地形对齐；可选联合动作头拟合地形专家，独立平地保持源学生行为。"""

import argparse
from dataclasses import asdict
import json
import shutil
import time

import torch

from env.commands import nominal_flat_training_config
from env.perception.runtime import VisualRuntime
from env.training_task import Go2Task
from rl.algorithms.perceptive_networks import PrivilegedCritic
from rl.algorithms.ppo import PPO
from rl.config import DistillConfig, TrainConfig
from rl.distillation import collect_sequences, merge_sequence_batches, update_sequences
from rl.run_io import append_csv, deployment_config, load_policy, new_run, save_checkpoint, save_json
from scripts.distill_reference_control import reference_visual


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', required=True)
    parser.add_argument('--flat-reference', required=True)
    parser.add_argument('--teacher', required=True)
    parser.add_argument('--num-envs', type=int, default=64)
    parser.add_argument('--free-envs', type=int, default=64)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--visual-steps', type=int, default=20)
    parser.add_argument('--burn-in', type=int, default=5)
    parser.add_argument('--sequence-batch', type=int, default=16)
    parser.add_argument('--learning-rate', type=float, default=2e-5)
    parser.add_argument('--reference-actions', action='store_true', help='用视觉专家的原相机/GRU动作标签代替高度标签')
    parser.add_argument('--control-lr', type=float, default=2e-5, help='reference-actions模式的动作MLP学习率')
    parser.add_argument('--teacher-mix', type=float, default=.5)
    parser.add_argument('--flat-weight', type=float, default=5.)
    parser.add_argument('--training-levels', type=int, nargs='+', default=[0,6,9,12])
    parser.add_argument('--seed', type=int, default=89998)
    args=parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    student, config, saved=load_policy(args.resume)
    reference, _, reference_saved=load_policy(args.flat_reference)
    teacher, _, teacher_saved=load_policy(args.teacher)
    assert student.mode==reference.mode=='student'
    assert teacher.mode==('student' if args.reference_actions else 'teacher')
    # 特征监督只适用于相同控制网络，避免把控制差异误当成视觉误差。
    for name, parameter in student.named_parameters():
        if not args.reference_actions and not name.startswith('visual_encoder.'):
            assert torch.equal(parameter, dict(reference.named_parameters())[name]), name
    student.requires_grad_(False)
    student.visual_encoder.requires_grad_(True)
    if args.reference_actions:
        student.actor.requires_grad_(True)
    reference.requires_grad_(False)
    teacher.requires_grad_(False)
    training=TrainConfig(**saved['train_config'])
    critic=PrivilegedCritic(training).to('cuda:0')
    critic.load_state_dict(saved['critic'])
    critic.requires_grad_(False)
    agent=PPO(actor=student, critic=critic, config=training, device='cuda:0')
    groups=[dict(params=student.visual_encoder.parameters(),lr=args.learning_rate)]
    if args.reference_actions:
        groups.append(dict(params=student.actor.parameters(),lr=args.control_lr))
    agent.actor_optimizer=torch.optim.Adam(groups)
    algorithm='visual_reference_action_supervision' if args.reference_actions else 'visual_terrain_alignment'
    # 动作监督和PPO的优化器均不能作为视觉优化器恢复。
    if saved.get('algorithm')==algorithm:
        agent.actor_optimizer.load_state_dict(saved['actor_optimizer'])
        agent.actor_optimizer.param_groups[0]['lr']=args.learning_rate
        if args.reference_actions:
            agent.actor_optimizer.param_groups[1]['lr']=args.control_lr
    distill=DistillConfig(burn_in=args.burn_in, visual_steps=args.visual_steps,
                         sequence_batch=args.sequence_batch, action_coef=1., latent_coef=.1,
                         freeze_common=True, held_terrain_guide=True)
    config.num_envs, config.seed=args.num_envs, args.seed
    config.randomization.enabled=False
    config.terrain.seed=56000
    folder=new_run(algorithm, config, training)
    metadata=dict(status='running', algorithm=algorithm, parent=args.resume,
                  arguments=vars(args), randomization_enabled=False)
    save_json(folder/'metadata.json', metadata)
    print('RUN_DIR='+str(folder), flush=True)
    tasks=[]
    try:
        route=Go2Task(config, curriculum=False)
        tasks.append(route)
        terrain=route.env.scene.terrain
        levels=torch.tensor(args.training_levels, device=route.device)
        terrain.terrain_levels[:]=levels[torch.arange(args.num_envs, device=route.device)%len(levels)]
        terrain.env_origins[:]=terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
        route.reset()
        free_config=nominal_flat_training_config(config, args.free_envs, args.seed+1)
        tasks.append(Go2Task(free_config, curriculum=False))
        save_json(folder/'task_configs.json', dict(route=asdict(config), free=asdict(free_config)))
        samplers=[(task, VisualRuntime(task, student.visual_encoder),
                   reference_visual(task, reference, reference_saved),
                   reference_visual(task,teacher,teacher_saved) if args.reference_actions else None) for task in tasks]
        deployment=deployment_config(route)
        began=time.monotonic()
        for iteration in range(1,args.iterations+1):
            batches=[]
            sampling={}
            for name,(task,visual,flat_visual,terrain_visual) in zip(('route','free'),samplers):
                batch,stats=collect_sequences(task,visual,teacher,student,
                    (args.visual_steps+args.burn_in)*5,args.teacher_mix,
                    held_terrain_guide=not args.reference_actions,flat_reference_visual=flat_visual,flat_weight=args.flat_weight,
                    terrain_reference_visual=terrain_visual,flat_reference_policy=reference)
                batches.append(batch)
                sampling.update({name+'_'+key:value for key,value in stats.items()})
            if any(task.numerical_failures for task in tasks):
                raise RuntimeError('仿真出现数值异常，停止本批视觉更新并保留记录')
            metrics=update_sequences(student,agent.actor_optimizer,merge_sequence_batches(batches),distill)
            row=dict(iteration=iteration,elapsed_seconds=time.monotonic()-began,
                     numerical_failures=sum(t.numerical_failures for t in tasks),**sampling,**metrics)
            append_csv(folder/'metrics.csv',row)
            print(json.dumps(row),flush=True)
            if iteration%25==0 or iteration==args.iterations:
                save_checkpoint(folder/'checkpoints/latest.pt',agent,iteration,config,training,'distill',
                    task=route,deployment=deployment,algorithm=algorithm,
                    source_ppo_iteration=saved['source_ppo_iteration'],
                    source_ppo_total_steps=saved['source_ppo_total_steps'],
                    source_visual_iteration=saved.get('source_visual_iteration',0)+iteration,
                    source_reference_control_updates=saved.get('source_reference_control_updates',0)+(iteration if args.reference_actions else 0),
                    flat_reference=args.flat_reference,teacher=args.teacher)
                shutil.copyfile(folder/'checkpoints/latest.pt',folder/f'checkpoints/iteration_{iteration}.pt')
        shutil.copyfile(folder/'checkpoints/latest.pt',folder/'checkpoints/final.pt')
        metadata.update(status='finished',iteration=iteration,numerical_failures=sum(t.numerical_failures for t in tasks))
        save_json(folder/'metadata.json',metadata)
    except BaseException as error:
        metadata.update(status='failed',error=str(error))
        save_json(folder/'metadata.json',metadata)
        raise
    finally:
        for task in tasks:
            task.close()


if __name__=='__main__':
    main()
