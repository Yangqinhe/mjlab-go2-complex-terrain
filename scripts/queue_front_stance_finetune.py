"""等待当前完整域随机化验收通过，再接续前腿站姿微调，绝不提前启动。"""

import json
from pathlib import Path
import subprocess
import sys
import time

from rl.run_io import ROOT, save_json


def main():
    pointer_path = ROOT/'outputs/active_student_training.json'
    pointer = json.loads(pointer_path.read_text())
    folder = Path(pointer['controller'])
    request_path = folder/'front_stance_finetune_queue.json'
    request = dict(status='waiting_for_full_domain_randomization',
                   prerequisite=pointer['status_file'], original_controller_pid=pointer['controller_pid'],
                   reason='用户确认GitHub student_final.pt前腿左右外八字；域随机化完成后再微调',
                   current_training_changed=False)
    save_json(request_path, request)
    print(f'QUEUED={request_path}', flush=True)
    while True:
        state = json.loads(Path(pointer['status_file']).read_text())
        if state['status'] in ('failed_needs_fix', 'interrupted', 'superseded_preserved'):
            request.update(status='blocked_by_domain_randomization', prerequisite_status=state['status'])
            save_json(request_path, request)
            return
        ready = (state['status']=='performance_passed_needs_delivery_review'
                 and state.get('strength')==1. and state.get('validation', {}).get('passed') is True)
        if ready:
            try:
                alive = b'scripts.train_student_continual' in Path(f"/proc/{pointer['controller_pid']}/cmdline").read_bytes()
            except FileNotFoundError:
                alive = False
            if not alive:
                break
        time.sleep(30)
    checkpoint = Path(state['checkpoint'])
    if not checkpoint.is_file():
        raise FileNotFoundError(f'已通过域随机化的检查点不存在：{checkpoint}')
    log = folder/'front_stance_finetune_controller.log'
    request.update(status='starting_front_stance_finetune', parent_checkpoint=str(checkpoint), log=str(log))
    save_json(request_path, request)
    command = [sys.executable, '-u', '-m', 'scripts.train_student_continual',
               '--initial-checkpoint', str(checkpoint), '--front-posture-finetune']
    with log.open('w') as stream:
        child = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        notification = None
        for line in child.stdout:
            stream.write(line)
            stream.flush()
            if line.startswith('CONTINUAL_DIR='):
                next_folder = Path(line.strip().split('=', 1)[1])
                new_pointer = dict(controller=str(next_folder), status_file=str(next_folder/'status.json'),
                    launch_log=str(log), controller_pid=child.pid, training_module='scripts.train_student_continual',
                    goal='完整域随机化下继续前腿站姿微调，保留六档路线、变速停车与60ms步态指标',
                    parent_domain_randomization=str(folder))
                save_json(pointer_path, new_pointer)
                request.update(status='front_stance_finetune_running', controller=str(next_folder), pid=child.pid)
                save_json(request_path, request)
                with (next_folder/'completion_watcher.log').open('w') as notify_log:
                    notification = subprocess.Popen([sys.executable, '-u', '-m', 'scripts.watch_student_training'],
                        cwd=ROOT, stdout=notify_log, stderr=subprocess.STDOUT)
        exit_code = child.wait()
    request.update(status='front_stance_finetune_controller_exited', exit_code=exit_code,
                   notification_pid=notification.pid if notification else None)
    save_json(request_path, request)


if __name__ == '__main__':
    main()
