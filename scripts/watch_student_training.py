"""监看持续训练的实际验收状态，停止后保存报告并通知当前电脑用户。"""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
from zoneinfo import ZoneInfo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pointer', default='outputs/active_student_training.json')
    args = parser.parse_args()
    pointer = json.loads(Path(args.pointer).read_text())
    folder = Path(pointer['controller'])
    status_path = Path(pointer['status_file'])
    pid = pointer['controller_pid']
    print(f'WATCHING={status_path}', flush=True)
    while True:
        try:
            current = json.loads(Path(args.pointer).read_text())
            if Path(current['controller']) != folder:
                return
            # 同一控制目录可在批次边界恢复；跟随最新PID，避免误报训练已停止。
            pointer = current
            pid = pointer['controller_pid']
            state = json.loads(status_path.read_text())
        except json.JSONDecodeError:
            # 控制器可能正在写状态文件，下一次读取完整内容。
            time.sleep(1)
            continue
        cmdline = Path(f'/proc/{pid}/cmdline')
        try:
            alive = pointer['training_module'].encode() in cmdline.read_bytes()
        except FileNotFoundError:
            alive = False
        passed = (state['status'] == 'performance_passed_needs_delivery_review'
                  and state.get('strength') == 1.
                  and state.get('validation', {}).get('passed') is True)
        failed = state['status'] in ('failed_needs_fix', 'interrupted', 'superseded_preserved')
        if (passed or failed or not alive) and not alive:
            break
        time.sleep(30)

    now = datetime.now(ZoneInfo('Asia/Hong_Kong')).isoformat(timespec='seconds')
    checkpoint = state.get('checkpoint') if passed else state.get('selected_checkpoint')
    title = 'Go2学生完整域随机化验收通过' if passed else 'Go2学生训练中断，尚未完成验收'
    message = ('训练已自动停止。路线、速度、停车及相机边界指标通过，视频与交付待复核。'
               if passed else f"当前状态：{state['status']}。请检查训练日志，不属于训练成功。")
    if passed and state.get('front_posture_finetune'):
        title = 'Go2前腿站姿微调验收通过'
        message = '前腿站宽与外展角指标、完整域随机化任务指标通过，训练已停止，视频待复核。'
    elif passed and (folder/'front_stance_finetune_queue.json').exists():
        message = '域随机化阶段通过并停止，已排队接续前腿站姿微调；最终微调策略尚未发布。'
    report = folder / ('训练验收通过通知.md' if passed else '训练中断通知.md')
    lines = [f'# {title}', '', f'时间：{now}', '', message, '',
             f"状态：`{state['status']}`", f'检查点：`{checkpoint}`',
             f'完整状态：`{status_path}`', f"当前日志：`{state.get('log')}`", '']
    if passed:
        lines += ['验收标准：完整强度随机化、六档固定120/120与独立120/120；'
                  '速度RMSE≤0.1m/s、连续腾空≤0.06秒及原步态指标；'
                  '33项变速停车与独立33项、20/35/50°相机边界。', '',
                  '这是仿真指标通过通知，尚未代替视频审看或实机验证。', '',
                  '前腿站姿阶段另要求：平均站宽24.4～32.4cm、P95≤34.4cm，前髋绝对角P95≤0.25rad。'
                  if state.get('front_posture_finetune') else '前腿站姿微调是否待接续，以排队状态为准。', '',
                  f"权重SHA256：`{hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()}`", '',
                  '验收报告：', '', '```json',
                  json.dumps(state['validation'], ensure_ascii=False, indent=2), '```']
    report.write_text('\n'.join(lines) + '\n')
    result = dict(time=now, passed=passed, report=str(report), checkpoint=checkpoint,
                  training_stopped=True, desktop_notification_sent=False)
    if shutil.which('notify-send'):
        try:
            completed = subprocess.run(['notify-send', '--app-name=Go2学生训练',
                '--urgency=critical', '--expire-time=0', title, message + f'\n报告：{report}'],
                capture_output=True, text=True, timeout=10)
            result['desktop_notification_sent'] = completed.returncode == 0
            if completed.returncode:
                result['notification_error'] = completed.stderr.strip()
        except subprocess.TimeoutExpired:
            result['notification_error'] = '桌面通知服务超时，结果报告已保存'
    (folder / 'completion_notification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
