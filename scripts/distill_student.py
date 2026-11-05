"""从已验收的地形教师训练视觉学生，按顺序监督GRU而非随机打散图像。"""

import argparse
import hashlib
from dataclasses import asdict
import json
import time

import torch

from env.training_task import Go2Task
from env.perception.runtime import VisualRuntime
from rl.config import DistillConfig, TrainConfig
from rl.algorithms.perceptive_networks import PerceptiveActor, PrivilegedCritic
from rl.algorithms.ppo import PPO
from rl.distillation import collect_sequences, update_sequences
from rl.run_io import (append_csv, deployment_config, load_policy, new_run,
                       restore_curriculum, save_checkpoint, save_json)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--teacher', required=True)
    parser.add_argument('--resume')
    parser.add_argument('--seed', type=int, help='采样与仿真初始化seed')
    parser.add_argument('--terrain-seed', type=int, help='地形几何seed，与采样seed分别记录')
    parser.add_argument('--num-envs', type=int, default=256)
    parser.add_argument('--iterations', type=int, default=200)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--warmup', type=int, default=50)
    parser.add_argument('--burn-in', type=int, default=10)
    parser.add_argument('--visual-steps', type=int, default=20)
    parser.add_argument('--sequence-batch', type=int, default=16)
    parser.add_argument('--visual-lr', type=float)
    parser.add_argument('--common-lr', type=float)
    parser.add_argument('--training-levels', type=int, nargs='+', help='固定混合档位，重复某档可增加其采样比例')
    parser.add_argument('--teacher-mix', type=float, help='采样时固定教师动作比例，评估仍只使用学生')
    parser.add_argument('--copy-teacher-common', action='store_true', help='恢复后将共享网络重置为教师参数')
    parser.add_argument('--freeze-common', action='store_true', help='本阶段只训练视觉编码器')
    args = parser.parse_args()
    torch.set_num_threads(1)
    teacher, env_config, checkpoint = load_policy(args.teacher, args.device)
    teacher.requires_grad_(False)
    env_config.num_envs = args.num_envs
    if args.seed is not None:
        env_config.seed = args.seed
        torch.manual_seed(args.seed)
    if args.terrain_seed is not None:
        env_config.terrain.seed = args.terrain_seed
    config = DistillConfig(num_envs=args.num_envs, burn_in=args.burn_in,
                           visual_steps=args.visual_steps, sequence_batch=args.sequence_batch,
                           training_levels=tuple(args.training_levels) if args.training_levels else None,
                           teacher_mix=args.teacher_mix, copy_teacher_common=args.copy_teacher_common,
                           freeze_common=args.freeze_common)
    saved = torch.load(args.resume, map_location=args.device, weights_only=False) if args.resume else None
    training = TrainConfig(**(saved or checkpoint)['train_config'])
    student = PerceptiveActor('student', training).to(args.device)
    student.copy_common(teacher)
    critic = PrivilegedCritic(training).to(args.device)
    critic.load_state_dict(checkpoint['critic'])
    agent = PPO(actor=student, critic=critic, config=training, device=args.device)
    common = [p for name,p in student.named_parameters() if not name.startswith('visual_encoder.')]
    agent.actor_optimizer = torch.optim.Adam([
        dict(params=student.visual_encoder.parameters(), lr=config.visual_lr),
        dict(params=common, lr=config.common_lr),
    ])
    start = 0
    curriculum_saved = checkpoint
    if saved:
        student.load_state_dict(saved['actor'])
        critic.load_state_dict(saved['critic'])
        # PPO为单参数组，蒸馏为视觉/共享两组；跨阶段保留权重并新建优化器。
        if saved['stage'] == 'distill':
            agent.actor_optimizer.load_state_dict(saved['actor_optimizer'])
            start = saved['iteration']
        curriculum_saved = saved
    if config.copy_teacher_common:
        student.copy_common(teacher)
        for parameter in common:
            agent.actor_optimizer.state.pop(parameter, None)
    # 恢复优化器后再覆盖显式学习率，并记录实际生效值。
    for group, name in zip(agent.actor_optimizer.param_groups, ('visual_lr', 'common_lr')):
        value = getattr(args, name)
        if value is not None:
            group['lr'] = value
        setattr(config, name, group['lr'])
    folder = new_run('distill', env_config, training)
    save_json(folder/'distillation.json', asdict(config))
    metadata = dict(teacher=args.teacher, parent=args.resume, status='running',
                    parent_stage=saved['stage'] if saved else None, seed=env_config.seed,
                    terrain_seed=env_config.terrain.seed)
    save_json(folder/'metadata.json', metadata)
    print(f'RUN_DIR={folder}', flush=True)
    task = Go2Task(env_config, args.device)
    model = task.env.sim.mj_model
    metadata['terrain_geometry_sha256'] = hashlib.sha256(
        model.geom_pos.tobytes()+model.geom_size.tobytes()).hexdigest()
    save_json(folder/'metadata.json', metadata)
    restore_curriculum(task, curriculum_saved)
    if config.training_levels is not None:
        # 固定混合档位保证最高档持续有数据；不因学生早期失败自动退回低档。
        levels = torch.tensor(config.training_levels, device=args.device)
        terrain = task.env.scene.terrain
        terrain.terrain_levels[:] = levels[torch.arange(args.num_envs, device=args.device) % len(levels)]
        terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
        task.curriculum = False
        task.reset()
    deployment = deployment_config(task)
    save_json(folder/'deployment.json', deployment)
    visual = VisualRuntime(task, student.visual_encoder)
    began = time.monotonic()
    try:
        for iteration in range(start+1, start+args.iterations+1):
            warm = iteration <= args.warmup
            for parameter in common:
                parameter.requires_grad_(not (warm or config.freeze_common))
            beta = (config.teacher_mix if config.teacher_mix is not None else
                    min(1., max(0., 1-(iteration-args.warmup)/100)))
            batch, sampling = collect_sequences(task, visual, teacher, student,
                (config.visual_steps+config.burn_in)*5, beta)
            metrics = update_sequences(student, agent.actor_optimizer, batch, config)
            row = dict(iteration=iteration, beta=beta, elapsed_seconds=time.monotonic()-began,
                       normal_captures=visual.normal_captures, reset_captures=visual.reset_captures,
                       numerical_failures=task.numerical_failures,
                       **sampling, **metrics)
            append_csv(folder/'metrics.csv', row)
            print(json.dumps(row), flush=True)
            if iteration % 25 == 0 or iteration == start+args.iterations:
                save_checkpoint(folder/'checkpoints/latest.pt', agent, iteration,
                                env_config, training, 'distill', teacher=args.teacher,
                                deployment=deployment, task=task)
        save_checkpoint(folder/'checkpoints/final.pt', agent, iteration,
                        env_config, training, 'distill', teacher=args.teacher,
                        deployment=deployment, task=task)
        print(f'DISTILLATION_FINISHED={folder}', flush=True)
        metadata.update(status='finished', iteration=iteration, numerical_failures=task.numerical_failures)
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
