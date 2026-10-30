"""仅发布通过完整独立评估的视觉学生；可等待训练流水线完成后自动上传Release。"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
from datetime import datetime

import torch

from rl.run_io import ROOT, save_json


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
    if (saved['stage'] != 'student_ppo' or not report.get('terrain_obstacles')
            or report['partial_evaluation'] or report['unique_layouts'] < 20
            or report['success_rate'] < .8 or report['vx_rmse'] > .1
            or Path(report['checkpoint']).resolve() != checkpoint.resolve()):
        raise RuntimeError('学生权重与独立评估未满足发布条件')
    delivery = folder/'delivery'
    delivery.mkdir(exist_ok=True)
    shutil.copyfile(checkpoint, delivery/'student_final.pt')
    shutil.copyfile(evaluation, delivery/'student_evaluation.json')
    shutil.copyfile(checkpoint.parent.parent/'config.json', delivery/'config.json')
    shutil.copyfile(checkpoint.parent.parent/'deployment.json', delivery/'deployment.json')
    # 从已通过评估的layout0案例单独重放，视频只记录一个回合，避免其他世界reset穿插。
    case = next(row for row in report['records'] if row['layout'] == 0 and row['success'])
    with (folder/'delivery_video.log').open('w') as stream:
        subprocess.run([str(ROOT/'run.sh'), 'python', '-m', 'scripts.record_video',
            '--checkpoint', str(checkpoint), '--num-envs', '1', '--seeds', str(case['seed']),
            '--level', str(report['level']), '--video', str(delivery/'student_low_stairs.mp4')],
            cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
    video_report_path = next(line.split('=', 1)[1]
        for line in (folder/'delivery_video.log').read_text().splitlines()
        if line.startswith('EVALUATION='))
    if not json.loads(Path(video_report_path).read_text())['records'][0]['success']:
        raise RuntimeError('单环境录像重放未成功，保留报告但不发布成功视频')
    shutil.copyfile(video_report_path, delivery/'video_evaluation.json')
    for log in folder.glob('student_level_*.log'):
        for line in log.read_text().splitlines():
            if line.startswith('EVALUATION='):
                shutil.copyfile(line.split('=', 1)[1], delivery/f'{log.stem}.json')
    subprocess.run([str(ROOT/'run.sh'), 'python', '-m', 'scripts.plot_training',
                    str(checkpoint.parent.parent/'metrics.csv')], cwd=ROOT, check=True)
    shutil.copyfile(checkpoint.parent.parent/'training_curves.png', delivery/'training_curves.png')
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
        f"固定vx=0.3m/s；独立低台阶混合路线{report['unique_layouts']}个布局，"
        f"通过率{report['success_rate']:.1%}，vx RMSE={report['vx_rmse']:.4f}m/s。\n\n"
        '包含平地、上下楼梯、静态碎石。中高台阶结果单独附JSON，'
        '不以低台阶通过率代表全部高度。域随机化关闭，尚未经实机验证。\n\n'
        '附件包含完整学生checkpoint、配置、关节/相机定义、独立评估、训练曲线、视频和SHA256。'
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
