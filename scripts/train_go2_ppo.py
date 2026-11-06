"""自写PPO的GPU教师/学生训练入口；按rollout保存最终观测再显式reset。"""

import argparse
import hashlib
import json
import math
import time

import torch

from env.config import EnvConfig
from env.training_task import Go2Task
from rl.config import TrainConfig
from rl.algorithms.perceptive_networks import PerceptiveActor, PrivilegedCritic
from rl.algorithms.ppo import PPO
from rl.run_io import (append_csv, deployment_config, env_from_dict, new_run,
                       restore_curriculum, save_checkpoint, save_json)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['teacher', 'student_ppo'], default='teacher')
    parser.add_argument('--num-envs', type=int, default=2048)
    parser.add_argument('--iterations', type=int, default=500)
    parser.add_argument('--resume')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--levels', type=int)
    parser.add_argument('--layouts', type=int)
    parser.add_argument('--horizon', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--save-interval', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--terrain-seed', type=int)
    terrain_mode = parser.add_mutually_exclusive_group()
    terrain_mode.add_argument('--flat', action='store_true')
    terrain_mode.add_argument('--complex', action='store_true')
    parser.add_argument('--actor-lr', type=float)
    parser.add_argument('--velocity-coef', type=float)
    parser.add_argument('--kl-interval', type=int)
    parser.add_argument('--step-height-min', type=float)
    parser.add_argument('--step-height-max', type=float)
    parser.add_argument('--action-scale', type=float)
    parser.add_argument('--height-scan-clip', type=float)
    parser.add_argument('--adaptive-lr', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--kl-stop-multiplier', type=float)
    parser.add_argument('--route-following', action='store_true', default=None)
    parser.add_argument('--walking-gait', action='store_true', default=None)
    parser.add_argument('--freeze-proprio', action=argparse.BooleanOptionalAction, default=None,
                        help='是否固定本体编码与速度头；恢复权重后也可显式解冻')
    parser.add_argument('--proprio-only', action='store_true', default=None,
                        help='训练仅本体对照：关闭相机，融合中的视觉64维恒为零')
    parser.add_argument('--target-action-std', type=float)
    parser.add_argument('--long-flight-penalty', type=float)
    parser.add_argument('--flight-penalty', type=float)
    parser.add_argument('--terrain-support-reward', type=float, help='复杂地形落足重叠系数，0保留原奖励')
    parser.add_argument('--terrain-support-downstairs-only', action=argparse.BooleanOptionalAction,
                        default=None, help='将落足重叠限制在下楼梯及其前后0.2m')
    parser.add_argument('--training-levels', type=int, nargs='+')
    args = parser.parse_args()
    if args.proprio_only and args.stage != 'student_ppo':
        parser.error('--proprio-only只用于学生PPO对照')
    torch.set_num_threads(1)
    torch.manual_seed(args.seed if args.seed is not None else 42)
    config = EnvConfig(num_envs=args.num_envs, seed=args.seed if args.seed is not None else 42)
    config.terrain.levels, config.terrain.layouts = args.levels or 8, args.layouts or 4
    training = TrainConfig()
    saved = torch.load(args.resume, map_location=args.device, weights_only=False) if args.resume else None
    if saved:
        config = env_from_dict(saved['env_config'])
        config.num_envs = args.num_envs
        training = TrainConfig(**saved['train_config'])
    if args.seed is not None:
        config.seed = args.seed
    if args.terrain_seed is not None:
        config.terrain.seed = args.terrain_seed
    if args.route_following is not None:
        config.route.enabled = args.route_following
    if args.walking_gait is not None:
        config.gait.enabled = args.walking_gait
    if args.long_flight_penalty is not None:
        config.gait.long_flight = args.long_flight_penalty
    if args.flight_penalty is not None:
        config.gait.flight = args.flight_penalty
    if args.terrain_support_reward is not None:
        config.gait.terrain_support = args.terrain_support_reward
    if args.terrain_support_downstairs_only is not None:
        config.gait.terrain_support_downstairs_only = args.terrain_support_downstairs_only
    if args.flat or args.complex:
        config.terrain.obstacles = not args.flat
    for name in ['step_height_min', 'step_height_max', 'levels', 'layouts']:
        value = getattr(args, name)
        if value is not None:
            setattr(config.terrain, name, value)
    for name in ['action_scale', 'height_scan_clip']:
        value = getattr(args, name)
        if value is not None:
            setattr(config, name, value)
    for name in ['actor_lr', 'velocity_coef', 'kl_interval', 'horizon', 'batch_size',
                 'adaptive_lr', 'kl_stop_multiplier', 'save_interval', 'freeze_proprio',
                 'target_action_std', 'proprio_only']:
        value = getattr(args, name)
        if value is not None:
            setattr(training, name, value)
    mode = 'teacher' if args.stage == 'teacher' else 'student'
    if args.training_levels is not None:
        training.training_levels = tuple(args.training_levels)
    actor, critic = PerceptiveActor(mode, training), PrivilegedCritic(training)
    if mode == 'student':
        if saved is None or saved['stage'] not in ['distill', 'student_ppo']:
            raise ValueError('学生PPO必须从已蒸馏学生权重启动')
        actor.visual_encoder.requires_grad_(False)
        if training.freeze_proprio:
            # 微调动作头时保留已验证的本体特征与速度估计，避免同时改变策略输入。
            actor.history.requires_grad_(False)
            actor.velocity_head.requires_grad_(False)
        training.actor_lr = args.actor_lr if args.actor_lr is not None else 1e-4
        training.critic_lr = 3e-4
    agent = PPO(actor=actor, critic=critic, config=training, device=args.device,
                actor_lr=training.actor_lr, critic_lr=training.critic_lr,
                gamma=training.gamma, lmbda=training.lmbda, epochs=training.epochs,
                epsilon=training.epsilon)
    start_iteration = 0
    total_steps = 0
    if saved:
        agent.actor.load_state_dict(saved['actor'])
        agent.critic.load_state_dict(saved['critic'])
        if saved['stage'] == args.stage:
            agent.actor_optimizer.load_state_dict(saved['actor_optimizer'])
            agent.critic_optimizer.load_state_dict(saved['critic_optimizer'])
            if training.proprio_only != saved['train_config'].get('proprio_only', False):
                agent.actor_optimizer.state.clear()
            start_iteration = saved['iteration']
            total_steps = saved.get('total_steps', saved['iteration']*saved['train_config']['horizon']
                                    *saved['env_config']['num_envs'])
            for group in agent.actor_optimizer.param_groups:
                if args.actor_lr is not None:
                    group['lr'] = training.actor_lr
            for group in agent.critic_optimizer.param_groups:
                group['lr'] = training.critic_lr
    if training.target_action_std is not None:
        # 仅在采样前重定探索幅度，保留各关节std比例；旧log_prob尚未采集。
        with torch.no_grad():
            current = agent.actor.log_std.exp().mean().item()
            agent.actor.log_std.add_(math.log(training.target_action_std/current))
        agent.actor_optimizer.state.pop(agent.actor.log_std, None)
    # Torch峰值覆盖环境构建与训练；整进程/整卡峰值由容量入口另行采样。
    torch.cuda.reset_peak_memory_stats(args.device)
    folder = new_run(args.stage, config, training)
    metadata = dict(parent=args.resume, device=args.device,
                    gpu=torch.cuda.get_device_name(args.device), status='running',
                    num_envs=config.num_envs, seed=config.seed, terrain_seed=config.terrain.seed,
                    proprio_only=training.proprio_only,
                    torch_peak_scope='environment_and_training')
    save_json(folder/'metadata.json', metadata)
    print(f'RUN_DIR={folder}', flush=True)
    initialization_begin = time.monotonic()
    task = Go2Task(config, args.device)
    model = task.env.sim.mj_model
    metadata['terrain_geometry_sha256'] = hashlib.sha256(
        model.geom_pos.tobytes()+model.geom_size.tobytes()).hexdigest()
    restore_curriculum(task, saved)
    if training.training_levels is not None:
        # 与蒸馏的固定混合档位一致，最高档失败时仍持续采样，不自动降档。
        levels = torch.tensor(training.training_levels, device=args.device)
        terrain = task.env.scene.terrain
        terrain.terrain_levels[:] = levels[torch.arange(args.num_envs, device=args.device) % len(levels)]
        terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
        task.curriculum = False
        task.reset()
    deployment = deployment_config(task)
    deployment['proprio_only'] = training.proprio_only
    save_json(folder/'deployment.json', deployment)
    visual = None
    if mode == 'student' and not training.proprio_only:
        from env.perception.runtime import VisualRuntime
        visual = VisualRuntime(task, agent.actor.visual_encoder)
    latent = torch.zeros(config.num_envs, 64, device=args.device) if training.proprio_only else None
    if saved and saved['stage'] == args.stage and args.seed is None:
        torch.set_rng_state(saved['torch_rng'].cpu())
        torch.cuda.set_rng_state_all([state.cpu() for state in saved['cuda_rng']])
    torch.cuda.synchronize(args.device)
    metadata['environment_initialization_seconds'] = time.monotonic()-initialization_begin
    save_json(folder/'metadata.json', metadata)
    begin = time.monotonic()
    try:
        for iteration in range(start_iteration+1, start_iteration+args.iterations+1):
            statistics = torch.zeros(6, device=args.device)
            finished, success_count, fall_count, return_sum, rmse_sum = 0, 0, 0, 0., 0.
            reward_sum = torch.zeros((), device=args.device)
            term_sums = {}
            for _ in range(training.horizon):
                states, critic_states = task.inputs(latent if visual is None else visual.latent)
                action, sample = agent.sample_batch(states, critic_states)
                reward, terminated, truncated, info = task.step(action)
                if visual is not None:
                    visual.update()
                _, next_critic = task.inputs()
                agent.store_batch(sample, reward, terminated | truncated, terminated, next_critic)
                reward_sum += reward.mean()
                for name, value in info['terms'].items():
                    term_sums[name] = term_sums.get(name, 0)+value.mean()
                statistics += torch.stack([info['velocity'].mean(), info['error'].mean(),
                                           info['progress'].mean(), info['height'].mean(),
                                           info['terms']['slip'].mean(), info['terms']['effort'].mean()])
                ids = (terminated | truncated).nonzero(as_tuple=False).flatten()
                if len(ids):
                    finished += len(ids)
                    success_count += info['success'][ids].sum().item()
                    fall_count += info['fall'][ids].sum().item()
                    return_sum += info['episode_return'][ids].sum().item()
                    rmse_sum += info['rmse'][ids].sum().item()
                    task.reset(ids, info['success'][ids])
                    if visual is not None:
                        visual.reset(ids)
            update_begin = time.monotonic()
            metrics = agent.update_batch()  # 返回的.item()指标已同步本次更新。
            update_seconds = time.monotonic()-update_begin
            total_steps += training.horizon*config.num_envs
            values = (statistics/training.horizon).tolist()
            row = dict(iteration=iteration, total_steps=total_steps,
                       ppo_update_seconds=update_seconds,
                       reward=reward_sum.item()/training.horizon, vx=values[0],
                       tracking_rmse=values[1]**.5, progress=values[2], base_height=values[3],
                       slip_penalty=values[4], effort_penalty=values[5], episodes=finished,
                       successes=success_count, falls=fall_count,
                       numerical_failures=task.numerical_failures,
                       episode_return=return_sum/max(finished, 1),
                       episode_rmse=rmse_sum/max(finished, 1),
                       mean_level=task.env.scene.terrain.terrain_levels.float().mean().item(),
                       elapsed_seconds=time.monotonic()-begin, **metrics,
                       **{f'reward_{key}':value.item()/training.horizon
                          for key,value in term_sums.items()})
            append_csv(folder/'metrics.csv', row)
            with (folder/'train.log').open('a') as stream:
                stream.write(json.dumps(row)+'\n')
            if iteration % 10 == 0 or iteration == start_iteration+1:
                print(json.dumps(row), flush=True)
            if iteration % training.save_interval == 0 or iteration == start_iteration+args.iterations:
                save_checkpoint(folder/'checkpoints/latest.pt', agent, iteration, config, training, args.stage,
                                deployment=deployment, task=task, total_steps=total_steps)
                save_checkpoint(folder/f'checkpoints/iteration_{iteration:06d}.pt',
                                agent, iteration, config, training, args.stage, deployment=deployment,
                                task=task, total_steps=total_steps)
        save_checkpoint(folder/'checkpoints/final.pt', agent, iteration, config, training, args.stage,
                        deployment=deployment, task=task, total_steps=total_steps)
        print(f'TRAINING_FINISHED={folder}', flush=True)
        metadata.update(status='finished', iteration=iteration, numerical_failures=task.numerical_failures,
                        policy_steps_this_run=training.horizon*args.iterations,
                        torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(args.device),
                        torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(args.device))
        if visual is not None:
            metadata.update(normal_captures=visual.normal_captures, reset_captures=visual.reset_captures,
                            depth_capture_calls=visual.sensor.capture_count)
        elif training.proprio_only:
            metadata.update(normal_captures=0, reset_captures=0, depth_capture_calls=0)
        save_json(folder/'metadata.json', metadata)
    except BaseException as error:
        metadata.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                        error=type(error).__name__)
        save_json(folder/'metadata.json', metadata)
        raise
    finally:
        task.close()


if __name__ == '__main__':
    main()
