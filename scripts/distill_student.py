"""从已验收的地形教师训练视觉学生，按顺序监督GRU而非随机打散图像。"""

import argparse
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
    parser.add_argument('--num-envs', type=int, default=256)
    parser.add_argument('--iterations', type=int, default=200)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--warmup', type=int, default=50)
    parser.add_argument('--burn-in', type=int, default=10)
    parser.add_argument('--visual-steps', type=int, default=20)
    parser.add_argument('--sequence-batch', type=int, default=16)
    args = parser.parse_args()
    torch.set_num_threads(1)
    teacher, env_config, checkpoint = load_policy(args.teacher, args.device)
    teacher.requires_grad_(False)
    env_config.num_envs = args.num_envs
    config = DistillConfig(num_envs=args.num_envs, burn_in=args.burn_in,
                           visual_steps=args.visual_steps, sequence_batch=args.sequence_batch)
    training = TrainConfig(**checkpoint['train_config'])
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
    if args.resume:
        saved = torch.load(args.resume, map_location=args.device, weights_only=False)
        student.load_state_dict(saved['actor'])
        agent.actor_optimizer.load_state_dict(saved['actor_optimizer'])
        start = saved['iteration']
        curriculum_saved = saved
    folder = new_run('distill', env_config, training)
    save_json(folder/'distillation.json', asdict(config))
    save_json(folder/'metadata.json', dict(teacher=args.teacher, parent=args.resume, status='running'))
    print(f'RUN_DIR={folder}', flush=True)
    task = Go2Task(env_config, args.device)
    restore_curriculum(task, curriculum_saved)
    deployment = deployment_config(task)
    save_json(folder/'deployment.json', deployment)
    visual = VisualRuntime(task, student.visual_encoder)
    began = time.monotonic()
    try:
        for iteration in range(start+1, start+args.iterations+1):
            warm = iteration <= args.warmup
            for parameter in common:
                parameter.requires_grad_(not warm)
            beta = min(1., max(0., 1-(iteration-args.warmup)/100))
            batch, sampling = collect_sequences(task, visual, teacher, student,
                (config.visual_steps+config.burn_in)*5, beta)
            metrics = update_sequences(student, agent.actor_optimizer, batch, config)
            row = dict(iteration=iteration, beta=beta, elapsed_seconds=time.monotonic()-began,
                       normal_captures=visual.normal_captures, reset_captures=visual.reset_captures,
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
        save_json(folder/'metadata.json', dict(teacher=args.teacher, parent=args.resume,
                  status='finished', iteration=iteration))
    except BaseException as error:
        save_json(folder/'metadata.json', dict(teacher=args.teacher, parent=args.resume,
                  status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                  error=type(error).__name__))
        raise
    finally:
        task.close()


if __name__ == '__main__':
    main()
