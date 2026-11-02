"""依次训练教师、蒸馏视觉学生和PPO微调；独立评估未达标时不交付成功权重。"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess

from rl.run_io import ROOT, save_json
from rl.config import STAIR_STAGES


def run_logged(folder, name, module, arguments, marker):
    log = folder/f'{name}.log'
    command = [str(ROOT/'run.sh'), 'python', '-m', module, *map(str, arguments)]
    state = json.loads((folder/'status.json').read_text())
    state.update(stage=name, log=str(log))
    save_json(folder/'status.json', state)
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
    parser.add_argument('--skip-pretraining', action='store_true', help='从复杂地形教师继续，不重复平地/低障碍课程')
    parser.add_argument('--max-step-height', type=float, default=.20,
                        choices=[height for height, _ in STAIR_STAGES])
    parser.add_argument('--teacher-iterations', type=int, default=500)
    parser.add_argument('--terrain-warmup-iterations', type=int, default=300)
    parser.add_argument('--teacher-rounds', type=int, default=3)
    parser.add_argument('--distill-iterations', type=int, default=200)
    parser.add_argument('--student-iterations', type=int, default=300)
    parser.add_argument('--student-rounds', type=int, default=3)
    parser.add_argument('--actor-lr', type=float)
    parser.add_argument('--horizon', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--adaptive-lr', action='store_true')
    parser.add_argument('--kl-stop-multiplier', type=float)
    args = parser.parse_args()
    if args.skip_pretraining and not args.teacher_start:
        parser.error('--skip-pretraining需要--teacher-start')
    stages = [(height, scale) for height, scale in STAIR_STAGES if height <= args.max_step_height]
    ppo_args = []
    for name in ['actor_lr', 'horizon', 'batch_size', 'kl_stop_multiplier']:
        value = getattr(args, name)
        if value is not None:
            ppo_args.extend(['--'+name.replace('_', '-'), value])
    if args.adaptive_lr:
        ppo_args.append('--adaptive-lr')
    folder = ROOT/'outputs'/f"{datetime.now():%Y%m%d_%H%M%S}_pipeline"
    folder.mkdir(parents=True)
    save_json(folder/'arguments.json', vars(args))
    save_json(folder/'curriculum.json', [dict(max_step_height=h, action_scale=s,
              levels=max(8, round((h-.03)/.01)+1)) for h,s in stages])
    state = dict(status='running', folder=str(folder), target_step_height=args.max_step_height)
    save_json(folder/'status.json', state)
    print(f'PIPELINE_DIR={folder}', flush=True)
    try:
        checkpoint = args.teacher_start
        if not checkpoint:
            warm = run_logged(folder, 'flat', 'scripts.train_go2_ppo',
                ['--flat', '--iterations', 300, '--actor-lr', .0001,
                 '--velocity-coef', .01, '--kl-interval', 1], 'TRAINING_FINISHED')
            checkpoint = warm/'checkpoints/final.pt'
        if not args.skip_pretraining:
            report, report_path = evaluate(folder, 'flat_eval', checkpoint)
            if not passed(report):
                raise RuntimeError(f'平地教师未达标：{report_path}')
            warm = run_logged(folder, 'terrain_warmup', 'scripts.train_go2_ppo',
                ['--resume', checkpoint, '--complex', '--num-envs', 2048,
                 '--step-height-min', .005, '--step-height-max', .03,
                 '--iterations', args.terrain_warmup_iterations], 'TRAINING_FINISHED')
            checkpoint = warm/'checkpoints/final.pt'
        for cap, scale in stages:
            levels = max(8, round((cap-.03)/.01)+1)
            label = f'{round(cap*100):02d}cm'
            for index in range(args.teacher_rounds):
                teacher = run_logged(folder, f'teacher_{label}_{index}', 'scripts.train_go2_ppo',
                    ['--resume', checkpoint, '--complex', '--num-envs', 2048, '--levels', levels,
                     '--step-height-min', .03, '--step-height-max', cap, '--action-scale', scale,
                     '--height-scan-clip', 2 if cap > .125 else 1,
                     '--iterations', args.teacher_iterations,
                     '--velocity-coef', .01, '--kl-interval', 1, *ppo_args], 'TRAINING_FINISHED')
                checkpoint = teacher/'checkpoints/final.pt'
                state.update(teacher=str(checkpoint), training_step_height=cap)
                save_json(folder/'status.json', state)
                low, low_path = evaluate(folder, f'teacher_{label}_low_{index}', checkpoint)
                high, high_path = evaluate(folder, f'teacher_{label}_high_{index}', checkpoint, levels-1)
                if passed(low) and passed(high):
                    state['validated_teacher_step_height'] = cap
                    save_json(folder/'status.json', state)
                    break
            else:
                raise RuntimeError(f'{label}教师未达标，不升难度、不启动蒸馏：{low_path}, {high_path}')
        student = run_logged(folder, 'distill', 'scripts.distill_student',
            ['--teacher', checkpoint, '--num-envs', 256,
             '--iterations', args.distill_iterations], 'DISTILLATION_FINISHED')
        checkpoint = student/'checkpoints/final.pt'
        evaluate(folder, 'distill_eval', checkpoint)
        validation_levels = sorted({0, *[round((height-.03)/(args.max_step_height-.03)*(levels-1))
                                         for height, _ in stages]})
        for index in range(args.student_rounds):
            student = run_logged(folder, f'student_{index}', 'scripts.train_go2_ppo',
                ['--stage', 'student_ppo', '--resume', checkpoint, '--num-envs', 256,
                 '--iterations', args.student_iterations, *ppo_args], 'TRAINING_FINISHED')
            checkpoint = student/'checkpoints/final.pt'
            state['student'] = str(checkpoint)
            save_json(folder/'status.json', state)
            reports = []
            for level in validation_levels:
                report, path = evaluate(folder, f'student_level_{level}_round_{index}', checkpoint, level)
                reports.append(str(path))
                if not passed(report):
                    break
            else:
                state['height_evaluations'] = reports
                report_path = reports[0]
                break
        else:
            raise RuntimeError(f'视觉学生全高度验收尚未达标：{path}')
        video = folder/'student_low_stairs.mp4'
        run_logged(folder, 'video', 'scripts.record_video',
            ['--checkpoint', checkpoint, '--num-envs', 1, '--seeds', 10005, '--video', video], 'EVALUATION')
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
