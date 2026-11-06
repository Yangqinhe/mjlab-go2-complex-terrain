"""用学生访问状态校准现有速度估计头；其余策略权重固定，推理仍不读取速度真值。"""

import argparse
from dataclasses import asdict
from pathlib import Path
import shutil
import time

import torch
import torch.nn.functional as F

from env.training_task import Go2Task
from env.perception.runtime import VisualRuntime
from rl.config import TrainConfig
from rl.run_io import (append_csv, deployment_config, load_policy, new_run,
                       restore_curriculum, save_json)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--num-envs', type=int, default=256)
    parser.add_argument('--iterations', type=int, default=20)
    parser.add_argument('--horizon', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=4096)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=.0001)
    parser.add_argument('--seed', type=int, default=70000)
    args = parser.parse_args()
    torch.set_num_threads(1)
    actor, config, saved = load_policy(args.checkpoint)
    if saved['stage'] != 'student_ppo' or actor.proprio_only:
        raise ValueError('校准起点须为视觉PPO学生')
    config.num_envs, config.seed = args.num_envs, args.seed
    training = TrainConfig(**saved['train_config'])
    actor.requires_grad_(False)
    actor.velocity_head.requires_grad_(True)
    optimizer = torch.optim.Adam(actor.velocity_head.parameters(), lr=args.lr)
    folder = new_run('velocity_calibration', config, training)
    source_folder = Path(args.checkpoint).resolve().parent.parent
    shutil.copyfile(source_folder/'metrics.csv', folder/'metrics.csv')
    source_config = source_folder/'ppo_source_config.json'
    shutil.copyfile(source_config if source_config.exists() else source_folder/'config.json',
                    folder/'ppo_source_config.json')
    save_json(folder/'calibration.json', vars(args))
    metadata = dict(status='running', parent=args.checkpoint,
                    updated_module='velocity_head', additional_ppo_updates=0)
    save_json(folder/'metadata.json', metadata)
    print(f'CALIBRATION_DIR={folder}', flush=True)
    task = Go2Task(config)
    try:
        restore_curriculum(task, saved)
        if training.training_levels is not None:
            levels = torch.tensor(training.training_levels, device=task.device)
            terrain = task.env.scene.terrain
            terrain.terrain_levels[:] = levels[torch.arange(args.num_envs, device=task.device) % len(levels)]
            terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
            task.curriculum = False
            task.reset()
        visual = VisualRuntime(task, actor.visual_encoder)
        save_json(folder/'deployment.json', deployment_config(task))
        begin = time.monotonic()
        for iteration in range(1, args.iterations+1):
            features, targets, masks = [], [], []
            for _ in range(args.horizon):
                with torch.no_grad():
                    states, _ = task.inputs(visual.latent)
                    features.append(actor.history(states[:, 45:495]).clone())
                    targets.append(task.robot.data.root_link_lin_vel_b.clone())
                    # reset后的自由落体和未填满的历史不用于速度标定。
                    masks.append(task.env.episode_length_buf*task.dt > .5)
                    mu, std = actor(states)
                    _, terminated, truncated, info = task.step(mu+std*torch.randn_like(mu))
                    visual.update()
                    ids = (terminated | truncated).nonzero().flatten()
                    if len(ids):
                        task.reset(ids, info['success'][ids])
                        visual.reset(ids)
            mask = torch.stack(masks).flatten()
            x = torch.stack(features).flatten(0, 1)[mask]
            y = torch.stack(targets).flatten(0, 1)[mask]
            with torch.no_grad():
                before = (actor.velocity_head(x)-y).square().mean(0).sqrt()
            for _ in range(args.epochs):
                for ids in torch.randperm(len(x), device=task.device).split(args.batch_size):
                    loss = F.mse_loss(2*actor.velocity_head(x[ids]), 2*y[ids])
                    if not torch.isfinite(loss):
                        raise RuntimeError('Non-finite velocity calibration loss')
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(actor.velocity_head.parameters(), 1.)
                    optimizer.step()
            with torch.no_grad():
                after = (actor.velocity_head(x)-y).square().mean(0).sqrt()
            row = dict(iteration=iteration, samples=len(x), elapsed_seconds=time.monotonic()-begin,
                       numerical_failures=task.numerical_failures,
                       **{f'before_{axis}':before[i].item() for i,axis in enumerate('xyz')},
                       **{f'after_{axis}':after[i].item() for i,axis in enumerate('xyz')})
            append_csv(folder/'velocity_metrics.csv', row)
            print(row, flush=True)
            if iteration % 5 == 0 or iteration == args.iterations:
                # 保留真实PPO计数；校准单独记录，不伪造新增PPO更新或复用旧actor动量。
                payload = dict(saved, actor=actor.state_dict(), env_config=asdict(config),
                               torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                               estimator_calibration=dict(iterations=iteration, horizon=args.horizon,
                                   num_envs=args.num_envs, parent=args.checkpoint),
                               calibration_optimizer=optimizer.state_dict())
                payload['actor_optimizer'] = dict(saved['actor_optimizer'], state={})
                torch.save(payload, folder/'checkpoints/latest.pt')
        torch.save(payload, folder/'checkpoints/final.pt')
        metadata.update(status='finished', calibration_iterations=args.iterations,
                        numerical_failures=task.numerical_failures)
        save_json(folder/'metadata.json', metadata)
        print(f'VELOCITY_CALIBRATION_FINISHED={folder}', flush=True)
    except BaseException as error:
        metadata.update(status='failed', error=str(error))
        save_json(folder/'metadata.json', metadata)
        raise
    finally:
        task.close()


if __name__ == '__main__':
    main()
