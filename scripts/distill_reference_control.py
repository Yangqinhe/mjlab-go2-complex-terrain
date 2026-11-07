"""冻结感知和本体编码，监督拟合平地/复杂地形参考动作，再交回PPO续训。"""

import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import time

import torch

from env.gait import flat_terrain_mask
from env.perception.config import DepthConfig
from env.perception.depth_sensor import DepthSensor
from env.perception.runtime import VisualRuntime
from env.training_task import Go2Task
from rl.algorithms.perceptive_networks import PrivilegedCritic
from rl.algorithms.ppo import PPO
from rl.config import TrainConfig
from rl.run_io import append_csv, deployment_config, load_policy, new_run, save_checkpoint, save_json


def reference_visual(task, policy, saved):
    camera=DepthConfig(**saved['env_config'].get('camera', {}))
    groups=[0,4] if saved['env_config']['terrain'].get('surroundings') else [0]
    return VisualRuntime(task, policy.visual_encoder,
                         DepthSensor(task, camera, geometry_groups=groups, randomize=False))


def flat_training_config(source, num_envs, seed):
    """保持平地审计的场景设置，用独立训练seed采集源策略行为。"""
    config=deepcopy(source)
    config.num_envs, config.seed=num_envs, seed
    config.terrain.seed=seed
    config.randomization.seed=200000+seed
    config.randomization.enabled=False
    config.camera.pitch_down_degrees=25.
    config.terrain.obstacles=False
    config.terrain.levels, config.terrain.layouts=2, 1
    config.terrain.width, config.terrain.flat_length, config.terrain.spawn_x=20., 20., 10.
    config.route.enabled=False
    config.command.mode='legacy'
    config.command.vx=(.3,.3)
    config.command.vy=config.command.wz=(0.,0.)
    return config


def fit_actions(actor, optimizer, states, targets, weights, args):
    total, count=0., 0
    for _ in range(args.epochs):
        for ids in torch.randperm(len(states), device=states.device).split(args.batch_size):
            error=(actor(states[ids])[0]-targets[ids]).square().mean(-1)
            weight=weights[ids]
            loss=(error*weight).sum()/weight.sum()
            if not torch.isfinite(loss):
                raise RuntimeError('参考动作监督损失出现非有限值')
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.actor.parameters(), 1.)
            optimizer.step()
            total+=loss.item()
            count+=1
    return total/count


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', required=True)
    parser.add_argument('--flat-reference', required=True)
    parser.add_argument('--terrain-reference', required=True)
    parser.add_argument('--num-envs', type=int, default=128)
    parser.add_argument('--free-envs', type=int, default=0)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--horizon', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=8192)
    parser.add_argument('--learning-rate', type=float, default=2e-5)
    parser.add_argument('--teacher-mix', type=float, default=.5)
    parser.add_argument('--flat-weight', type=float, default=1.)
    parser.add_argument('--training-levels', type=int, nargs='+', default=[0,6,9,12])
    parser.add_argument('--seed', type=int, default=89991)
    args=parser.parse_args()
    if not 0 <= args.teacher_mix <= 1:
        parser.error('teacher-mix应在0到1之间')
    if args.flat_weight<=0:
        parser.error('flat-weight必须大于0')
    if args.free_envs<0:
        parser.error('free-envs不能为负数')
    torch.set_num_threads(1)
    actor, config, saved=load_policy(args.resume)
    flat, _, flat_saved=load_policy(args.flat_reference)
    rough, _, rough_saved=load_policy(args.terrain_reference)
    assert actor.mode==flat.mode==rough.mode=='student'
    actor.requires_grad_(False)
    actor.actor.requires_grad_(True)
    flat.requires_grad_(False)
    rough.requires_grad_(False)
    preserve_source=Path(args.resume).resolve()==Path(args.flat_reference).resolve()
    training=TrainConfig(**saved['train_config'])
    critic=PrivilegedCritic(training).to('cuda:0')
    critic.load_state_dict(saved['critic'])
    critic.requires_grad_(False)
    # 复用检查点格式；本阶段仅MSE更新动作MLP，不采集PPO优势或更新critic。
    agent=PPO(actor=actor, critic=critic, config=training, device='cuda:0')
    agent.actor_optimizer=torch.optim.Adam(actor.actor.parameters(), lr=args.learning_rate)
    config.num_envs, config.seed=args.num_envs, args.seed
    config.randomization.enabled=False
    config.terrain.seed=56000
    ppo_iteration=saved.get('source_ppo_iteration', saved['iteration']) if saved['stage']=='distill' else saved['iteration']
    ppo_steps=saved.get('source_ppo_total_steps', saved.get('total_steps',0)) if saved['stage']=='distill' else saved.get('total_steps',0)
    folder=new_run('reference_control_distill', config, training)
    metadata=dict(status='running', algorithm='reference_action_supervision', parent=args.resume,
                  source_ppo_iteration=ppo_iteration, source_visual_iteration=saved.get('source_visual_iteration'),
                  arguments=vars(args), flat_reference_is_source=preserve_source,
                  numerical_failures=0, randomization_enabled=False)
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
        free_config=flat_training_config(config, args.free_envs, args.seed+1) if args.free_envs else None
        if free_config:
            tasks.append(Go2Task(free_config, curriculum=False))
        save_json(folder/'task_configs.json', dict(route=asdict(config), free=asdict(free_config) if free_config else None))
        samplers=[]
        for task in tasks:
            views=[VisualRuntime(task, actor.visual_encoder), reference_visual(task, flat, flat_saved)]
            if task is route:
                views.append(reference_visual(task, rough, rough_saved))
            samplers.append((task,views))
        hips=[i for i,name in enumerate(route.robot.joint_names) if name.endswith('_hip_joint')]
        hip_limit=config.training_reward.front_hip_deadband/config.action_scale
        deployment=deployment_config(route)
        save_json(folder/'deployment.json', deployment)
        torch.manual_seed(args.seed)
        began=time.monotonic()
        for iteration in range(1, args.iterations+1):
            states, targets, weights=[], [], []
            episodes=successes=highest_episodes=highest_successes=0
            free_episodes=free_successes=0
            flat_fraction=0.
            flat_action_error=0.
            for _ in range(args.horizon):
                with torch.no_grad():
                    for task,views in samplers:
                        visual,flat_visual=views[:2]
                        state,_=task.inputs(visual.latent)
                        flat_state,_=task.inputs(flat_visual.latent)
                        local_x=task.robot.data.root_link_pos_w[:,0]-task.env.scene.env_origins[:,0]+task.config.terrain.spawn_x
                        is_flat=flat_terrain_mask(task, local_x)
                        target=flat(flat_state)[0]
                        if task is route:
                            rough_state,_=task.inputs(views[2].latent)
                            target=torch.where(is_flat[:,None], target, rough(rough_state)[0])
                        # 自参考保留完整源动作；外部参考沿用原有平地髋目标限制。
                        if not preserve_source:
                            straight=is_flat & (task.command()[:,2].abs()<.1)
                            target[:,hips]=torch.where(straight[:,None], target[:,hips].clamp(-hip_limit,hip_limit), target[:,hips])
                        student_mu=actor(state)[0]
                        flat_action_error+=((student_mu-target).square().mean(-1)*is_flat).sum().item()
                        action=args.teacher_mix*target+(1-args.teacher_mix)*student_mu
                        states.append(state.clone())
                        targets.append(target.clone())
                        weights.append(1+(args.flat_weight-1)*is_flat.float())
                        flat_fraction+=is_flat.sum().item()
                        _, terminated, truncated, info=task.step(action)
                        for runtime in views:
                            runtime.update()
                        ids=(terminated|truncated).nonzero(as_tuple=False).flatten()
                        if len(ids):
                            episodes+=len(ids)
                            successes+=int(info['success'][ids].sum())
                            if task is route:
                                highest=ids[terrain.terrain_levels[ids]==max(args.training_levels)]
                                highest_episodes+=len(highest)
                                highest_successes+=int(info['success'][highest].sum())
                            else:
                                free_episodes+=len(ids)
                                free_successes+=int(info['success'][ids].sum())
                            task.reset(ids)
                            for runtime in views:
                                runtime.reset(ids)
            loss=fit_actions(actor, agent.actor_optimizer, torch.cat(states), torch.cat(targets), torch.cat(weights), args)
            row=dict(iteration=iteration, action_loss=loss, teacher_mix=args.teacher_mix, flat_weight=args.flat_weight,
                     flat_reference_fraction=flat_fraction/(args.horizon*(args.num_envs+args.free_envs)),
                     flat_action_mse=flat_action_error/max(flat_fraction,1.),
                     episodes=episodes, successes=successes, free_episodes=free_episodes, free_successes=free_successes,
                     highest_level=max(args.training_levels), highest_episodes=highest_episodes,
                     highest_successes=highest_successes, numerical_failures=sum(t.numerical_failures for t in tasks),
                     elapsed_seconds=time.monotonic()-began)
            append_csv(folder/'metrics.csv', row)
            with (folder/'train.log').open('a') as stream:
                stream.write(json.dumps(row)+'\n')
            print(json.dumps(row), flush=True)
            if iteration%25==0 or iteration==args.iterations:
                save_checkpoint(folder/'checkpoints/latest.pt', agent, iteration, config, training, 'distill',
                    task=route, deployment=deployment, source_ppo_iteration=ppo_iteration,
                    source_ppo_total_steps=ppo_steps, source_visual_iteration=saved.get('source_visual_iteration'),
                    source_reference_control_updates=iteration, supervised_total_steps=iteration*args.horizon*(args.num_envs+args.free_envs),
                    free_training_envs=args.free_envs,
                    flat_reference=args.flat_reference, terrain_reference=args.terrain_reference,
                    algorithm='reference_action_supervision')
                shutil.copyfile(folder/'checkpoints/latest.pt', folder/f'checkpoints/iteration_{iteration}.pt')
        shutil.copyfile(folder/'checkpoints/latest.pt', folder/'checkpoints/final.pt')
        metadata.update(status='finished', iteration=iteration, numerical_failures=sum(t.numerical_failures for t in tasks))
        save_json(folder/'metadata.json', metadata)
        print('CONTROL_DISTILLATION_FINISHED='+str(folder), flush=True)
    except BaseException as error:
        metadata.update(status='failed', error=str(error), numerical_failures=sum(t.numerical_failures for t in tasks))
        save_json(folder/'metadata.json', metadata)
        raise
    finally:
        for task in tasks:
            task.close()


if __name__=='__main__':
    main()
