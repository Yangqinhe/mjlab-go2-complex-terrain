"""仅发布通过完整独立评估的视觉学生；可等待训练流水线完成后自动上传Release。"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
from datetime import datetime

import torch

from rl.run_io import ROOT, save_json
from rl.config import STAIR_STAGES
from env.gait import gait_passed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pipeline', type=Path, required=True)
    parser.add_argument('--wait', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    folder = args.pipeline.resolve()
    while True:
        state = json.loads((folder/'status.json').read_text())
        if state['status'] != 'running' or not args.wait:
            break
        time.sleep(30)
    if state['status'] != 'finished':
        raise RuntimeError(f"流水线尚未成功完成，不发布：{state}")
    checkpoint, evaluation = Path(state['checkpoint']), Path(state['evaluation'])
    report = json.loads(evaluation.read_text())
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if (saved['stage'] != 'student_ppo' or saved['train_config'].get('proprio_only', False)
            or report.get('proprio_only', False) or report.get('visual_ablation', 'none') != 'none'
            or not report.get('terrain_obstacles')
            or not saved['env_config'].get('gait', {}).get('enabled') or not gait_passed(report)
            or report['partial_evaluation'] or report['unique_layouts'] < 20
            or report['success_rate'] < .8 or report['vx_rmse'] > .1
            or Path(report['checkpoint']).resolve() != checkpoint.resolve()):
        raise RuntimeError('学生权重与独立评估未满足发布条件')
    height_paths = state.get('height_evaluations', [])
    height_reports = [json.loads(Path(path).read_text()) for path in height_paths]
    target = state.get('target_step_height', saved['env_config']['terrain']['step_height_max'])
    required = {.03, *[height for height, _ in STAIR_STAGES if height <= target]}
    verified = {round(r['stair_height'], 4) for r in height_reports}
    if (not height_reports or max(r['stair_height'] for r in height_reports) < target-1e-6
            or not required.issubset(verified)
            or any(r.get('proprio_only', False) or r.get('visual_ablation', 'none') != 'none'
                   or r['partial_evaluation'] or r['unique_layouts'] < 20
                   or not gait_passed(r)
                   or r['success_rate'] < .8 or r['vx_rmse'] > .1
                   or r.get('route_override') is not None
                   or r.get('route_following', False) != saved['env_config'].get('route', {}).get('enabled', False)
                   or Path(r['checkpoint']).resolve() != checkpoint.resolve()
                   for r in height_reports)):
        raise RuntimeError('目标最高台阶及分高度评估尚未全部通过，不发布最终权重')
    delivery = folder/'delivery'
    delivery.mkdir(exist_ok=True)
    shutil.copyfile(checkpoint, delivery/'student_final.pt')
    shutil.copyfile(evaluation, delivery/'student_evaluation.json')
    shutil.copyfile(checkpoint.parent.parent/'config.json', delivery/'config.json')
    shutil.copyfile(checkpoint.parent.parent/'deployment.json', delivery/'deployment.json')
    for name in ['calibration.json', 'velocity_metrics.csv', 'ppo_source_config.json']:
        path = checkpoint.parent.parent/name
        if path.exists():
            shutil.copyfile(path, delivery/name)
    if state.get('teacher'):
        shutil.copyfile(state['teacher'], delivery/'teacher_final.pt')
    # 从已通过评估的layout0案例单独重放，视频只记录一个回合，避免其他世界reset穿插。
    for demo in sorted(height_reports, key=lambda r:r['stair_height']):
        height_cm = round(demo['stair_height']*100)
        label = 'low' if demo['level'] == 0 else f'{height_cm}cm'
        case = next(row for row in demo['records'] if row['layout'] == 0 and row['success'])
        log = folder/f'delivery_video_{label}.log'
        with log.open('w') as stream:
            subprocess.run([str(ROOT/'run.sh'), 'python', '-m', 'scripts.record_video',
                '--checkpoint', str(checkpoint), '--num-envs', '1', '--seeds', str(case['seed']),
                '--level', str(demo['level']), '--video', str(delivery/f'student_{label}_stairs.mp4')],
                cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        video_report_path = next(line.split('=', 1)[1]
            for line in log.read_text().splitlines() if line.startswith('EVALUATION='))
        video_report = json.loads(Path(video_report_path).read_text())
        if (not video_report['records'][0]['success'] or video_report['vx_rmse'] > .1
                or not gait_passed(video_report)):
            raise RuntimeError(f'{label}录像重放未通过路线/速度/步态检查，保留报告但不发布')
        shutil.copyfile(video_report_path, delivery/f'video_{label}_evaluation.json')
        subprocess.run([str(ROOT/'run.sh'), 'python', '-m', 'scripts.plot_gait',
                        str(delivery/f'student_{label}_stairs.gait.csv')], cwd=ROOT, check=True)
    for path, height_report in zip(height_paths, height_reports):
        shutil.copyfile(path, delivery/f"student_height_{round(height_report['stair_height']*100):02d}cm.json")
    # 中间快照也可交付，曲线必须止于该权重的实际迭代，不能混入后续更新。
    with (checkpoint.parent.parent/'metrics.csv').open() as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames
        rows = [row for row in reader if int(row['iteration']) <= saved['iteration']]
    with (delivery/'training_metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    subprocess.run([str(ROOT/'run.sh'), 'python', '-m', 'scripts.plot_training',
                    str(delivery/'training_metrics.csv')], cwd=ROOT, check=True)
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    manifest = dict(source_revision=revision, checkpoint=str(checkpoint),
                    evaluation=str(evaluation), sha256={})
    for path in sorted(delivery.iterdir()):
        if path.name != 'manifest.json':
            manifest['sha256'][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    save_json(delivery/'manifest.json', manifest)
    notes = folder/'release_notes.md'
    notes.write_text(
        '# Go2 176维视觉本体策略\n\n'
        f"固定vx=0.3m/s；单级最高{target*100:.0f}cm，各验收高度均通过独立完整回合评估。\n\n"
        f"其中低台阶混合路线{report['unique_layouts']}个布局，"
        f"通过率{report['success_rate']:.1%}，vx RMSE={report['vx_rmse']:.4f}m/s。\n\n"
        '包含平地、上下楼梯、静态碎石。所有验收高度结果单独附JSON，'
        '域随机化关闭，尚未经实机验证。\n\n'
        '路线跟踪由外部横向位置/航向反馈生成wz指令，vx固定0.3m/s；'
        '部署需提供相对路线的里程计定位，配置见deployment.json。\n\n'
        '附件包含完整学生checkpoint、配置、关节/相机定义、独立评估、训练曲线、视频和SHA256。'
        '每档同时通过接触步态及整条路线腾空检查；视频附50Hz四足接触CSV和步态曲线。'
        f'网络版本perceptive_176_v1，源码提交{revision}。\n')
    if args.prepare_only:
        print(f'DELIVERY={delivery}', flush=True)
        return
    tag = f'trained-176-{datetime.now():%Y%m%d-%H%M%S}'
    subprocess.run(['gh', 'release', 'create', tag, *map(str, sorted(delivery.iterdir())),
                    '--repo', 'Yangqinhe/mjlab-go2-complex-terrain', '--target', revision,
                    '--title', 'Go2视觉本体176维固定速度策略', '--notes-file', str(notes)],
                   cwd=ROOT, check=True)
    state['release'] = f'https://github.com/Yangqinhe/mjlab-go2-complex-terrain/releases/tag/{tag}'
    save_json(folder/'status.json', state)
    print(f"PUBLISHED={state['release']}", flush=True)


if __name__ == '__main__':
    main()
