"""对验收后的同一视觉策略做闭环深度干预，并与教师、本体基线生成配对地形对照。"""

import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from env.gait import gait_passed
from rl.run_io import ROOT, save_json
from scripts.train_pipeline import passed, run_logged


def read_json(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pipeline', required=True, type=Path)
    parser.add_argument('--teacher-audit', required=True, type=Path)
    parser.add_argument('--baseline-summary', required=True, type=Path)
    args = parser.parse_args()
    state = read_json(args.pipeline/'status.json')
    if state['status'] != 'finished':
        raise ValueError('先完成视觉学生的六档选择和独立seed验收，再生成最终对照')
    checkpoint = Path(state['checkpoint']).resolve()
    selection = [read_json(path) for path in state['best_student_reports']]
    fresh = [read_json(path) for path in state['height_evaluations']]
    for reports in [selection, fresh]:
        if ({round(r['stair_height']*100) for r in reports} != {3, 9, 12, 15, 18, 20}
                or any(not passed(r) or Path(r['checkpoint']).resolve() != checkpoint for r in reports)):
            raise ValueError('六档完整验收或权重对应关系不正确')
    selection_seeds = {seed for r in selection for seed in r['seeds']}
    fresh_seeds = {seed for r in fresh for seed in r['seeds']}
    if selection_seeds & fresh_seeds:
        raise ValueError('独立复核seed不能与选择seed重叠')
    teacher = read_json(args.teacher_audit)
    baseline = read_json(args.baseline_summary)
    if (Path(teacher['checkpoint']).resolve() != Path(state['teacher']).resolve()
            or hashlib.sha256(Path(teacher['checkpoint']).read_bytes()).hexdigest() != teacher['checkpoint_sha256']):
        raise ValueError('教师参考权重与审计不一致')
    folder = ROOT/'outputs'/f'{datetime.now():%Y%m%d_%H%M%S}_perception_comparison'
    folder.mkdir()
    status = dict(status='running', checkpoint=str(checkpoint),
                  checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(), rows=[])
    save_json(folder/'status.json', status)
    save_json(folder/'baseline_training_summary.json', baseline)
    save_json(folder/'teacher_reference_audit.json', teacher)
    print(f'COMPARISON_DIR={folder}', flush=True)
    rows = []

    def add(variant, path, expected_checkpoint, mode='none', proprio=False):
        report = read_json(path)
        cases = {(r['seed'], r['layout']) for r in report['records']}
        expected = {(seed, layout) for seed in range(10000, 10005) for layout in range(4)}
        if (report['partial_evaluation'] or cases != expected or report['unique_layouts'] != 20
                or Path(report['checkpoint']).resolve() != Path(expected_checkpoint).resolve()
                or report.get('visual_ablation', 'none') != mode
                or report.get('proprio_only', False) != proprio or report.get('route_override') is not None):
            raise ValueError(f'{variant}不是同一套完整的20个地形案例')
        if variant in ('Teacher', 'Vision') and not passed(report):
            raise ValueError(f'{variant}的参考结果未通过')
        height = round(report['stair_height']*100)
        name = f'{variant.lower()}_{height:02d}cm.json'
        shutil.copyfile(path, folder/name)
        row = dict(variant=variant, height_cm=height, episodes=len(report['records']),
                   success_rate=report['success_rate'], vx_rmse=report['vx_rmse'],
                   max_flight_seconds=report['gait']['all_max_flight'],
                   route_and_gait_metrics_passed=(report['success_rate'] >= .8
                       and report['vx_rmse'] <= .1 and gait_passed(report)), report=name,
                   source_report=str(Path(path).resolve()))
        rows.append(row)
        current = read_json(folder/'status.json')
        current['rows'] = rows
        save_json(folder/'status.json', current)

    try:
        for height, level in [(3, 0), (20, 17)]:
            add('Teacher', teacher['heights'][str(height/100)]['report'], teacher['checkpoint'])
            item = next(r for r in baseline['results'] if r['height_cm'] == height)
            add('Proprio', item['report'], baseline['checkpoint'], proprio=True)
            path = next(p for p in state['best_student_reports'] if round(read_json(p)['stair_height']*100) == height)
            add('Vision', path, checkpoint)
            for mode, variant in [('far', 'Far'), ('swap_frames', 'Swap')]:
                path = run_logged(folder, f'{mode}_{height}cm', 'scripts.evaluate', [
                    '--checkpoint', checkpoint, '--num-envs', 4, '--level', level,
                    '--seeds', 10000, 10001, 10002, 10003, 10004,
                    '--visual-ablation', mode], 'EVALUATION')
                add(variant, path, checkpoint, mode=mode)
        with (folder/'comparison.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        variants = ['Teacher', 'Proprio', 'Vision', 'Far', 'Swap']
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
        for column, height in enumerate([3, 20]):
            group = [next(r for r in rows if r['height_cm'] == height and r['variant'] == v) for v in variants]
            for axis, key, label in [(axes[0, column], 'success_rate', 'Full-route success rate'),
                                     (axes[1, column], 'vx_rmse', 'Mean per-episode vx RMSE (m/s)')]:
                axis.bar(variants, [r[key] for r in group], color=['#8793a5', '#dba24b', '#327db4', '#be6666', '#62a493'])
                axis.set_title(f'{height} cm stairs: 20 complete episodes')
                axis.set_ylabel(label)
                axis.grid(axis='y', alpha=.2)
                if key == 'success_rate':
                    axis.set_ylim(0, 1.05)
        fig.savefig(folder/'comparison.png', dpi=200)
        plt.close(fig)
        result = dict(checkpoint=str(checkpoint), checkpoint_sha256=status['checkpoint_sha256'],
                      rows=rows, seeds=list(range(10000, 10005)), layouts=list(range(4)),
                      interpretation='Far and Swap change only depth input of the same selected policy. '
                      'Proprio is a separately fine-tuned fixed-budget baseline with different training history; '
                      'this comparison does not establish the optimal performance of blind locomotion.')
        save_json(folder/'comparison.json', result)
        status = read_json(folder/'status.json')
        status['status'] = 'finished'
        save_json(folder/'status.json', status)
        print(f'COMPARISON_FINISHED={folder}', flush=True)
    except BaseException as error:
        status = read_json(folder/'status.json')
        status.update(status='failed', error=str(error))
        save_json(folder/'status.json', status)
        raise


if __name__ == '__main__':
    main()
