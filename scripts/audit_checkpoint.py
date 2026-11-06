"""直接验收已保存的学生权重；六档选模与新seed均通过后才允许准备交付。"""

import argparse
from datetime import datetime
import json
import hashlib
from pathlib import Path

import torch

from env.gait import MAX_FLIGHT_SECONDS
from rl.config import STAIR_STAGES
from rl.run_io import ROOT, save_json
from scripts.train_pipeline import evaluate, passed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--pipeline', type=Path, required=True, help='提供教师和训练来源的流水线')
    parser.add_argument('--fresh-seed', type=int, required=True, help='尚未用于选模的5个连续seed起点')
    parser.add_argument('--selection-reports', nargs='+', type=Path,
                        help='复用同一权重已有的固定seed六档完整报告，仍完整执行新seed复核')
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if (saved['stage'] != 'student_ppo' or saved['train_config'].get('proprio_only', False)
            or not saved['env_config']['gait']['enabled']):
        raise ValueError('这里只验收启用步态约束的PPO视觉学生')
    if set(range(args.fresh_seed, args.fresh_seed+5)) & set(range(10000, 10005)):
        raise ValueError('新seed不能与固定选模seed重叠')
    terrain = saved['env_config']['terrain']
    target = terrain['step_height_max']
    levels = sorted({0, *[round((h-.03)/(target-.03)*(terrain['levels']-1))
                         for h, _ in STAIR_STAGES if h <= target]})
    cached = {}
    for path in args.selection_reports or []:
        report = json.loads(path.read_text())
        cases = {(r['seed'], r['layout']) for r in report['records']}
        expected = {(seed, layout) for seed in range(10000, 10005) for layout in range(4)}
        if (Path(report['checkpoint']).resolve() != checkpoint
                or report['iteration'] != saved['iteration'] or not report['walking_gait']
                or report.get('route_override') is not None or not report.get('terrain_obstacles')
                or report.get('route_following', False) != saved['env_config'].get('route', {}).get('enabled', False)
                or cases != expected
                or len(report['records']) != 20 or not passed(report)):
            raise ValueError(f'已有选择报告与该权重、固定20案例或当前验收不符：{path}')
        if report['level'] in cached:
            raise ValueError('同一档位不能提供重复选择报告')
        cached[report['level']] = (report, path.resolve())
    if cached and set(cached) != set(levels):
        raise ValueError('复用报告必须覆盖全部六档')
    source = json.loads((args.pipeline/'status.json').read_text())
    folder = ROOT/'outputs'/f'{datetime.now():%Y%m%d_%H%M%S}_checkpoint_audit'
    folder.mkdir()
    state = dict(folder=str(folder), source_pipeline=str(args.pipeline.resolve()),
                 student=str(checkpoint), teacher=source['teacher'],
                 target_step_height=target, status='running', fresh_seed=args.fresh_seed,
                 max_flight_seconds=MAX_FLIGHT_SECONDS,
                 checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                 reused_selection_reports={str(level):dict(path=str(path),
                     sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                     for level, (_, path) in cached.items()})
    save_json(folder/'status.json', state)
    print(f'AUDIT_DIR={folder}', flush=True)
    try:
        # 先测最高档便于拒绝明显退化的快照；通过时仍须覆盖全部六档。
        order = [levels[-1], *levels[:-1]]
        for phase, seeds in [('selection', list(range(10000, 10005))),
                             ('fresh', list(range(args.fresh_seed, args.fresh_seed+5)))]:
            paths = {}
            for level in order:
                if phase == 'selection' and cached:
                    report, path = cached[level]
                else:
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
