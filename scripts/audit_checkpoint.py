"""直接验收已保存的学生权重；六档选模与新seed均通过后才允许准备交付。"""

import argparse
from datetime import datetime
import json
from pathlib import Path

import torch

from rl.config import STAIR_STAGES
from rl.run_io import ROOT, save_json
from scripts.train_pipeline import evaluate, passed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--pipeline', type=Path, required=True, help='提供教师和训练来源的流水线')
    parser.add_argument('--fresh-seed', type=int, required=True, help='尚未用于选模的5个连续seed起点')
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if saved['stage'] != 'student_ppo' or not saved['env_config']['gait']['enabled']:
        raise ValueError('这里只验收启用步态约束的PPO视觉学生')
    if set(range(args.fresh_seed, args.fresh_seed+5)) & set(range(10000, 10005)):
        raise ValueError('新seed不能与固定选模seed重叠')
    terrain = saved['env_config']['terrain']
    target = terrain['step_height_max']
    levels = sorted({0, *[round((h-.03)/(target-.03)*(terrain['levels']-1))
                         for h, _ in STAIR_STAGES if h <= target]})
    source = json.loads((args.pipeline/'status.json').read_text())
    folder = ROOT/'outputs'/f'{datetime.now():%Y%m%d_%H%M%S}_checkpoint_audit'
    folder.mkdir()
    state = dict(folder=str(folder), source_pipeline=str(args.pipeline.resolve()),
                 student=str(checkpoint), teacher=source['teacher'],
                 target_step_height=target, status='running', fresh_seed=args.fresh_seed)
    save_json(folder/'status.json', state)
    print(f'AUDIT_DIR={folder}', flush=True)
    try:
        # 先测最高档便于拒绝明显退化的快照；通过时仍须覆盖全部六档。
        order = [levels[-1], *levels[:-1]]
        for phase, seeds in [('selection', list(range(10000, 10005))),
                             ('fresh', list(range(args.fresh_seed, args.fresh_seed+5)))]:
            paths = {}
            for level in order:
                report, path = evaluate(folder, f'{phase}_{level}', checkpoint, level, seeds=seeds)
                paths[level] = str(path)
                state.update(stage=f'{phase}_{level}', **{f'{phase}_evaluations': dict(paths)})
                if not passed(report):
                    state.update(status='rejected', rejected_evaluation=str(path))
                    save_json(folder/'status.json', state)
                    print(f'AUDIT_REJECTED={path}', flush=True)
                    return
                save_json(folder/'status.json', state)
            if phase == 'selection':
                state['best_student_reports'] = [paths[level] for level in levels]
            else:
                state.update(height_evaluations=[paths[level] for level in levels],
                             evaluation=paths[0], checkpoint=str(checkpoint), status='finished')
        save_json(folder/'status.json', state)
        print(f'AUDIT_PASSED={folder}', flush=True)
    except BaseException as error:
        state.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                     error=str(error))
        save_json(folder/'status.json', state)
        raise


if __name__ == '__main__':
    main()
