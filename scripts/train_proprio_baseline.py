"""从视觉学生起点单独重训仅本体PPO基线，并保存固定预算后的六档比较报告。"""

import argparse
from datetime import datetime
import json
from pathlib import Path

import torch

from rl.config import STAIR_STAGES
from rl.run_io import ROOT, save_json
from scripts.train_pipeline import evaluate, run_logged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--num-envs', type=int, default=256)
    parser.add_argument('--seed', type=int, default=67100)
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if saved['stage'] != 'student_ppo' or saved['train_config'].get('proprio_only', False):
        raise ValueError('基线起点须为视觉PPO学生，不能重复计算已训练基线的预算')
    terrain = saved['env_config']['terrain']
    maximum = terrain['step_height_max']
    levels = sorted({0, *[round((height-.03)/(maximum-.03)*(terrain['levels']-1))
                         for height, _ in STAIR_STAGES if height <= maximum]})
    folder = ROOT/'outputs'/f'{datetime.now():%Y%m%d_%H%M%S}_proprio_baseline'
    folder.mkdir()
    state = dict(status='running', source_checkpoint=str(checkpoint),
                 additional_ppo_updates=args.iterations, evaluations={},
                 comparison='proprio-only fine-tuning from a visual warm start; not training from scratch')
    save_json(folder/'status.json', state)
    print(f'BASELINE_DIR={folder}', flush=True)
    try:
        run = run_logged(folder, 'train_proprio', 'scripts.train_go2_ppo', [
            '--stage', 'student_ppo', '--resume', checkpoint, '--proprio-only',
            '--num-envs', args.num_envs, '--iterations', args.iterations,
            '--seed', args.seed, '--terrain-seed', terrain['seed'], '--save-interval', 25,
            '--actor-lr', saved['train_config']['actor_lr']], 'TRAINING_FINISHED')
        state = json.loads((folder/'status.json').read_text())
        state['checkpoint'] = str(run/'checkpoints/final.pt')
        save_json(folder/'status.json', state)
        for level in levels:
            report, path = evaluate(folder, f'baseline_level_{level}', state['checkpoint'], level)
            assert report['proprio_only']
            state = json.loads((folder/'status.json').read_text())
            state['evaluations'][str(level)] = str(path)
            save_json(folder/'status.json', state)
        state['status'] = 'finished_comparison'
        save_json(folder/'status.json', state)
        print(f'BASELINE_COMPARISON={folder}', flush=True)
    except BaseException as error:
        state = json.loads((folder/'status.json').read_text())
        state.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error=str(error))
        save_json(folder/'status.json', state)
        raise


if __name__ == '__main__':
    main()
