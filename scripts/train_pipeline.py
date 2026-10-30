"""依次训练教师、蒸馏视觉学生和PPO微调；独立评估未达标时不交付成功权重。"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess

from rl.run_io import ROOT, save_json


def run_logged(folder, name, module, arguments, marker):
    log = folder/f'{name}.log'
    command = [str(ROOT/'run.sh'), 'python', '-m', module, *map(str, arguments)]
    print(f'STAGE={name} LOG={log}', flush=True)
    with log.open('w') as stream:
        subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
    for line in reversed(log.read_text().splitlines()):
        if line.startswith(marker+'='):
            return Path(line.split('=', 1)[1])
    raise RuntimeError(f'{name}缺少完成标记，检查{log}')


def evaluate(folder, name, checkpoint, level=0, video=None):
    args = ['--checkpoint', checkpoint, '--num-envs', 4,
            '--seeds', 10000, 10001, 10002, 10003, 10004, '--level', level]
    if video:
        args += ['--video', video]
    path = run_logged(folder, name, 'scripts.evaluate', args, 'EVALUATION')
    report = json.loads(path.read_text())
    print(json.dumps(dict(stage=name, success_rate=report['success_rate'],
                          vx_rmse=report['vx_rmse'], unique_layouts=report['unique_layouts'])), flush=True)
    return report, path


def passed(report):
    return (not report['partial_evaluation'] and report['unique_layouts'] >= 20
            and report['success_rate'] >= .8 and report['vx_rmse'] <= .1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--teacher-start', help='已有平地教师checkpoint，跳过平地预训练')
    parser.add_argument('--teacher-iterations', type=int, default=500)
    parser.add_argument('--teacher-rounds', type=int, default=3)
    parser.add_argument('--distill-iterations', type=int, default=200)
    parser.add_argument('--student-iterations', type=int, default=300)
    parser.add_argument('--student-rounds', type=int, default=3)
    args = parser.parse_args()
    folder = ROOT/'outputs'/f"{datetime.now():%Y%m%d_%H%M%S}_pipeline"
    folder.mkdir(parents=True)
    save_json(folder/'arguments.json', vars(args))
    state = dict(status='running', folder=str(folder))
    save_json(folder/'status.json', state)
    print(f'PIPELINE_DIR={folder}', flush=True)
    try:
        checkpoint = args.teacher_start
        if not checkpoint:
            warm = run_logged(folder, 'flat', 'scripts.train_go2_ppo',
                ['--flat', '--iterations', 300, '--actor-lr', .0001,
                 '--velocity-coef', .01, '--kl-interval', 1], 'TRAINING_FINISHED')
            checkpoint = warm/'checkpoints/final.pt'
        report, report_path = evaluate(folder, 'flat_eval', checkpoint)
        if not passed(report):
            raise RuntimeError(f'平地教师未达标：{report_path}')
        for index in range(args.teacher_rounds):
            teacher = run_logged(folder, f'teacher_{index}', 'scripts.train_go2_ppo',
                ['--resume', checkpoint, '--complex', '--num-envs', 2048,
                 '--iterations', args.teacher_iterations, '--actor-lr', .0001,
                 '--velocity-coef', .01, '--kl-interval', 1], 'TRAINING_FINISHED')
            checkpoint = teacher/'checkpoints/final.pt'
            state['teacher'] = str(checkpoint)
            save_json(folder/'status.json', state)
            report, report_path = evaluate(folder, f'teacher_eval_{index}', checkpoint)
            if passed(report):
                break
        else:
            raise RuntimeError(f'复杂地形教师尚未达标，不启动蒸馏：{report_path}')
        student = run_logged(folder, 'distill', 'scripts.distill_student',
            ['--teacher', checkpoint, '--num-envs', 256,
             '--iterations', args.distill_iterations], 'DISTILLATION_FINISHED')
        checkpoint = student/'checkpoints/final.pt'
        evaluate(folder, 'distill_eval', checkpoint)
        for index in range(args.student_rounds):
            student = run_logged(folder, f'student_{index}', 'scripts.train_go2_ppo',
                ['--stage', 'student_ppo', '--resume', checkpoint, '--num-envs', 256,
                 '--iterations', args.student_iterations], 'TRAINING_FINISHED')
            checkpoint = student/'checkpoints/final.pt'
            state['student'] = str(checkpoint)
            save_json(folder/'status.json', state)
            report, report_path = evaluate(folder, f'student_eval_{index}', checkpoint)
            if passed(report):
                break
        else:
            raise RuntimeError(f'视觉学生尚未达标：{report_path}')
        # 独立报告中高难度结果，不能用低台阶的通过率代表全部地形。
        for level in [3, 7]:
            evaluate(folder, f'student_level_{level}', checkpoint, level=level)
        video = folder/'student_low_stairs.mp4'
        run_logged(folder, 'video', 'scripts.record_video',
            ['--checkpoint', checkpoint, '--seeds', 10005, '--video', video], 'EVALUATION')
        state.update(status='finished', checkpoint=str(checkpoint), evaluation=str(report_path),
                     video=str(video))
        save_json(folder/'status.json', state)
    except BaseException as error:
        state.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                     error=str(error))
        save_json(folder/'status.json', state)
        raise


if __name__ == '__main__':
    main()
