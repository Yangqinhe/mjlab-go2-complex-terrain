"""分档测试真实视觉PPO容量及组件耗时；容量结果与策略行走验收分开记录。"""

import argparse
import csv
from datetime import datetime
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import time

import torch

from rl.run_io import ROOT, save_json


def component_profile(checkpoint, num_envs, seed, terrain_seed, output):
    """独立进程的启动位置微基准；真实复杂地形接触负载由完整PPO测试覆盖。"""
    import torch
    from env.training_task import Go2Task
    from env.perception.runtime import VisualRuntime
    from rl.algorithms.perceptive_networks import PrivilegedCritic
    from rl.config import TrainConfig
    from rl.run_io import load_policy, restore_curriculum

    torch.set_num_threads(1)
    started = time.monotonic()
    actor, config, saved = load_policy(checkpoint)
    config.num_envs, config.seed, config.terrain.seed = num_envs, seed, terrain_seed
    critic = PrivilegedCritic(TrainConfig(**saved['train_config'])).cuda().eval()
    critic.load_state_dict(saved['critic'])
    task = Go2Task(config, curriculum=False)
    try:
        restore_curriculum(task, saved)
        visual = VisualRuntime(task, actor.visual_encoder)
        torch.cuda.synchronize()
        result = dict(num_envs=num_envs, initialization_seconds=time.monotonic()-started,
                      scope='component microbenchmark at course start; not route validation')

        def measure(function, count=20):
            for _ in range(3):
                function()
            torch.cuda.synchronize()
            begin = time.monotonic()
            for _ in range(count):
                function()
            torch.cuda.synchronize()
            return (time.monotonic()-begin)/count

        with torch.no_grad():
            states, _ = task.inputs(visual.latent)
            action = actor(states)[0]
            result['physics_pd_policy_step_seconds'] = measure(lambda: task.env.step(action))
            result['observation_seconds'] = measure(lambda: task.inputs(visual.latent))
            result['depth_capture_seconds'] = measure(visual.sensor.capture)
            visual.pair[:, 1].copy_(visual.sensor.capture())
            p = task.proprio()
            result['cnn_gru_seconds'] = measure(lambda: actor.visual_encoder(visual.pair, p, visual.hidden))
            states, critic_states = task.inputs(visual.latent)
            result['actor_seconds'] = measure(lambda: actor(states))
            result['critic_seconds'] = measure(lambda: critic(critic_states))
            result.update(depth_pair_shape=list(visual.pair.shape), action_shape=list(action.shape),
                          fusion_dim=actor.actor[0].in_features)
        save_json(output, result)
    finally:
        task.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--num-envs', type=int, nargs='+', default=[16, 64, 256, 512, 1024, 2048])
    parser.add_argument('--iterations', type=int, default=20)
    parser.add_argument('--horizon', type=int, default=128)
    parser.add_argument('--seed', type=int, default=65000)
    parser.add_argument('--terrain-seed', type=int, default=65000)
    parser.add_argument('--gpu', default=os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0])
    # 独立进程防止组件微基准的CUDA上下文影响下一档整卡显存测量。
    parser.add_argument('--profile-envs', type=int)
    parser.add_argument('--profile-output', type=Path)
    args = parser.parse_args()
    if args.profile_envs is not None:
        component_profile(args.checkpoint, args.profile_envs, args.seed, args.terrain_seed, args.profile_output)
        return
    checkpoint = args.checkpoint.resolve()
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if saved['stage'] != 'student_ppo':
        raise ValueError('这里只测真实视觉PPO容量')
    if saved['train_config'].get('proprio_only', False):
        raise ValueError('本体基线不能作为视觉容量验收权重')
    folder = ROOT/'outputs'/f'{datetime.now():%Y%m%d_%H%M%S}_visual_capacity'
    folder.mkdir()
    state = dict(status='running', checkpoint=str(checkpoint),
                 checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(), results=[],
                 policy_seconds_per_world=args.iterations*args.horizon*.02,
                 scope='route_length_load' if args.iterations*args.horizon*.02 >= 45 else 'short_smoke',
                 software={name:version(name) for name in ['torch', 'mujoco', 'mujoco-warp', 'mjlab', 'warp-lang']})
    info = subprocess.check_output(['nvidia-smi', '-i', args.gpu,
        '--query-gpu=name,uuid,memory.total,driver_version', '--format=csv,noheader,nounits'], text=True)
    state['gpu'] = dict(zip(['name', 'uuid', 'total_memory_mib', 'driver_version'],
                           [value.strip() for value in next(csv.reader(info.splitlines()))]))
    state['baseline_gpu_memory_mib'] = float(subprocess.check_output([
        'nvidia-smi', '-i', args.gpu, '--query-gpu=memory.used',
        '--format=csv,noheader,nounits'], text=True).strip())
    processes = subprocess.check_output(['nvidia-smi',
        '--query-compute-apps=gpu_uuid,pid,used_memory', '--format=csv,noheader,nounits'], text=True)
    state['baseline_compute_processes'] = [dict(pid=int(pid.strip()), memory_mib=memory.strip())
        for uuid, pid, memory in csv.reader(processes.splitlines()) if uuid.strip() == state['gpu']['uuid']]
    state['memory_scope'] = 'Whole GPU including existing background processes; no unrelated process stopped'
    save_json(folder/'arguments.json', {k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()})
    save_json(folder/'status.json', state)
    print(f'CAPACITY_DIR={folder}', flush=True)
    try:
        for num_envs in args.num_envs:
            for divisor in [2, 4, 8]:
                batch_size = max(1, num_envs*args.horizon//divisor)
                trial = folder/f'envs_{num_envs}_batch_{batch_size}'
                trial.mkdir()
                command = [str(ROOT/'run.sh'), 'python', '-m', 'scripts.train_go2_ppo',
                    '--stage', 'student_ppo', '--resume', str(checkpoint), '--num-envs', str(num_envs),
                    '--iterations', str(args.iterations), '--horizon', str(args.horizon),
                    '--batch-size', str(batch_size), '--seed', str(args.seed), '--terrain-seed', str(args.terrain_seed),
                    # 保留来源的本体冻结、探索幅度与地形采样，避免低估联合训练显存。
                    '--actor-lr', str(saved['train_config']['actor_lr'])]
                state.update(stage=trial.name, log=str(trial/'training.log'))
                save_json(folder/'status.json', state)
                with (trial/'memory_mib.csv').open('w') as memory, (trial/'training.log').open('w') as log:
                    monitor = subprocess.Popen(['nvidia-smi', '-i', args.gpu, '--query-gpu=memory.used',
                        '--format=csv,noheader,nounits', '-lms', '100'], stdout=memory, stderr=subprocess.STDOUT)
                    try:
                        completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                    finally:
                        monitor.terminate()
                        monitor.wait()
                text = (trial/'training.log').read_text()
                samples = []
                for line in (trial/'memory_mib.csv').read_text().splitlines():
                    try:
                        samples.append(float(line))
                    except ValueError:
                        pass
                result = dict(num_envs=num_envs, batch_size=batch_size, exit_code=completed.returncode,
                              gpu_peak_mib=max(samples, default=None), trial=str(trial))
                if completed.returncode:
                    result['status'] = 'oom' if 'out of memory' in text.lower() else 'failed'
                    state['results'].append(result)
                    save_json(folder/'status.json', state)
                    if result['status'] == 'oom':
                        continue  # 先缩小minibatch，再确认该环境规模是否容纳得下。
                    break
                run = Path(next(line.split('=', 1)[1] for line in text.splitlines()
                                if line.startswith('TRAINING_FINISHED=')))
                metadata = json.loads((run/'metadata.json').read_text())
                with (run/'metrics.csv').open() as stream:
                    rows = list(csv.DictReader(stream))
                captures = args.horizon*args.iterations//5
                overflow = [line for line in text.splitlines()
                            if 'overflow' in line.lower() and any(word in line.lower() for word in ['contact', 'constraint'])]
                verified = (metadata['numerical_failures'] == 0 and metadata['normal_captures'] == captures
                            and metadata['depth_capture_calls'] >= captures+1 and bool(samples)
                            and any(int(row['actor_steps']) > 0 for row in rows) and not overflow)
                result.update(status='passed' if verified else 'unverified', training_run=str(run),
                    training_config=json.loads((run/'config.json').read_text())['train'],
                    metadata=metadata, contact_overflow_messages=overflow,
                    transitions_per_second=num_envs*args.horizon*args.iterations/float(rows[-1]['elapsed_seconds']),
                    optimizer_update_seconds_mean=sum(float(row['ppo_update_seconds']) for row in rows)/len(rows))
                if verified:
                    profile_command = [str(ROOT/'run.sh'), 'python', '-m', 'scripts.benchmark_visual_capacity',
                        '--checkpoint', str(checkpoint), '--profile-envs', str(num_envs),
                        '--profile-output', str(trial/'components.json'), '--seed', str(args.seed),
                        '--terrain-seed', str(args.terrain_seed)]
                    with (trial/'components.log').open('w') as log:
                        subprocess.run(profile_command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
                    result['components'] = json.loads((trial/'components.json').read_text())
                    assert result['components']['action_shape'] == [num_envs, 12]
                    assert result['components']['depth_pair_shape'] == [num_envs, 2, 48, 64]
                    assert result['components']['fusion_dim'] == 176
                state['results'].append(result)
                save_json(folder/'status.json', state)
                print(json.dumps({k:result[k] for k in ['num_envs', 'batch_size', 'status', 'gpu_peak_mib']}), flush=True)
                break
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == state['checkpoint_sha256']
        passed = [r['num_envs'] for r in state['results'] if r['status'] == 'passed']
        state.update(status='finished', max_verified_envs=max(passed, default=0))
        save_json(folder/'status.json', state)
        save_json(folder/'report.json', state)
        print(f'CAPACITY_REPORT={folder / "report.json"}', flush=True)
    except BaseException as error:
        state.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error=str(error))
        save_json(folder/'status.json', state)
        raise


if __name__ == '__main__':
    main()
