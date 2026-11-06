"""依次训练教师、蒸馏视觉学生和PPO微调；独立评估未达标时不交付成功权重。"""

import argparse
from datetime import datetime
import json
from itertools import count
from pathlib import Path
import subprocess

from rl.run_io import ROOT, save_json
from rl.config import STAIR_STAGES
from env.gait import gait_passed, gait_error


def run_logged(folder, name, module, arguments, marker):
    log = folder/f'{name}.log'
    if log.exists():
        log = folder/f'{name}_{datetime.now():%Y%m%d_%H%M%S}.log'
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


def evaluate(folder, name, checkpoint, level=0, video=None, seeds=None):
    args = ['--checkpoint', checkpoint, '--num-envs', 4,
            '--seeds', *(seeds or [10000, 10001, 10002, 10003, 10004]), '--level', level]
    if video:
        args += ['--video', video]
    path = run_logged(folder, name, 'scripts.evaluate', args, 'EVALUATION')
    report = json.loads(path.read_text())
    print(json.dumps(dict(stage=name, success_rate=report['success_rate'],
                          vx_rmse=report['vx_rmse'], unique_layouts=report['unique_layouts'])), flush=True)
    return report, path


def route_passed(report, min_success=.8, max_rmse=.1):
    return (not report.get('proprio_only', False) and report.get('visual_ablation', 'none') == 'none'
            and not report['partial_evaluation'] and report['unique_layouts'] >= 20
            and report['success_rate'] >= min_success and report['vx_rmse'] <= max_rmse)


def passed(report):
    return route_passed(report) and (not report.get('walking_gait') or gait_passed(report))


def selection_score(reports, gait_penalty=.1):
    """硬验收不变；训练候选兼顾进展和步态，避免回退到始终停住的模型。"""
    error = sum(gait_error(report) for report in reports)
    success = [report['success_rate'] for report in reports]
    return (sum(passed(report) for report in reports), min(success)-gait_penalty*error,
            sum(success), -error, -max(report['vx_rmse'] for report in reports))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--teacher-start', help='已有平地教师checkpoint，跳过平地预训练')
    parser.add_argument('--student-start', help='已有视觉学生checkpoint，跳过重复蒸馏')
    parser.add_argument('--resume-pipeline', type=Path, help='从此流水线最近已通过的教师档恢复')
    parser.add_argument('--skip-pretraining', action='store_true', help='从复杂地形教师继续，不重复平地/低障碍课程')
    parser.add_argument('--max-step-height', type=float, default=.20,
                        choices=[height for height, _ in STAIR_STAGES])
    parser.add_argument('--teacher-iterations', type=int, default=500)
    parser.add_argument('--terrain-warmup-iterations', type=int, default=300)
    parser.add_argument('--teacher-rounds', type=int, default=3, help='0表示持续训练至验收通过')
    parser.add_argument('--distill-iterations', type=int, default=200)
    parser.add_argument('--student-iterations', type=int, default=300)
    parser.add_argument('--student-num-envs', type=int, default=256, help='视觉学生PPO并行数，容量实测后再扩大')
    parser.add_argument('--student-rounds', type=int, default=3, help='0表示持续训练至验收通过')
    parser.add_argument('--student-actor-lr', type=float, help='单独设置学生学习率')
    parser.add_argument('--student-fixed-lr', action='store_true', help='学生关闭自适应学习率')
    parser.add_argument('--student-freeze-proprio', action='store_true', help='学生微调时固定本体特征和速度估计')
    parser.add_argument('--student-action-std', type=float, help='学生每批开始时的平均探索标准差')
    parser.add_argument('--student-long-flight-penalty', type=float, help='学生连续腾空超过宽限时间的惩罚')
    parser.add_argument('--student-flight-penalty', type=float, help='学生每个无支撑控制周期的基础惩罚')
    parser.add_argument('--student-continue-routes', action='store_true',
                        help='各档至少半数走完且RMSE≤0.2时连续更新；最佳候选另存，最终验收不变')
    parser.add_argument('--student-training-levels', type=int, nargs='+', help='学生固定混合档位，重复项作为采样权重')
    parser.add_argument('--student-terrain-seed-base', type=int, help='学生每批地形seed=此值+轮号')
    parser.add_argument('--actor-lr', type=float)
    parser.add_argument('--horizon', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--save-interval', type=int)
    parser.add_argument('--adaptive-lr', action='store_true')
    parser.add_argument('--kl-stop-multiplier', type=float)
    parser.add_argument('--route-following', action='store_true')
    parser.add_argument('--walking-gait', action='store_true')
    parser.add_argument('--flat-only', action='store_true', help='平地步态通过后保存并结束，供录像检查')
    parser.add_argument('--gait-selection-penalty', type=float, default=.1,
                        help='只用于失败候选选模，不改变步态验收门槛')
    args = parser.parse_args()
    if args.skip_pretraining and not args.teacher_start:
        parser.error('--skip-pretraining需要--teacher-start')
    stages = [(height, scale) for height, scale in STAIR_STAGES if height <= args.max_step_height]
    ppo_args = []
    for name in ['actor_lr', 'horizon', 'batch_size', 'kl_stop_multiplier', 'save_interval']:
        value = getattr(args, name)
        if value is not None:
            ppo_args.extend(['--'+name.replace('_', '-'), value])
    if args.adaptive_lr:
        ppo_args.append('--adaptive-lr')
    if args.route_following:
        ppo_args.append('--route-following')
    if args.walking_gait:
        ppo_args.append('--walking-gait')
    folder = args.resume_pipeline.resolve() if args.resume_pipeline else ROOT/'outputs'/f"{datetime.now():%Y%m%d_%H%M%S}_pipeline"
    folder.mkdir(parents=True, exist_ok=bool(args.resume_pipeline))
    arguments = vars(args).copy()
    arguments['resume_pipeline'] = str(args.resume_pipeline) if args.resume_pipeline else None
    save_json(folder/('resume_arguments.json' if args.resume_pipeline else 'arguments.json'), arguments)
    save_json(folder/'curriculum.json', [dict(max_step_height=h, action_scale=s,
              levels=max(8, round((h-.03)/.01)+1)) for h,s in stages])
    state = (json.loads((folder/'status.json').read_text()) if args.resume_pipeline else
             dict(folder=str(folder), target_step_height=args.max_step_height))
    state.update(status='running')
    state.pop('error', None)
    save_json(folder/'status.json', state)
    print(f'PIPELINE_DIR={folder}', flush=True)
    try:
        checkpoint = state['teacher'] if args.resume_pipeline else args.teacher_start
        if not checkpoint and args.walking_gait:
            for index in count():
                warm = run_logged(folder, f'flat_walk_{index}', 'scripts.train_go2_ppo',
                    ['--flat', '--num-envs', 2048, '--iterations', args.teacher_iterations,
                     '--seed', 71000+index, *(['--resume', checkpoint] if checkpoint else []),
                     *ppo_args], 'TRAINING_FINISHED')
                checkpoint = warm/'checkpoints/final.pt'
                state['teacher'] = str(checkpoint)
                save_json(folder/'status.json', state)
                report, report_path = evaluate(folder, f'flat_walk_eval_{index}', checkpoint)
                if passed(report):
                    break
            if args.flat_only:
                state.update(status='flat_ready', checkpoint=str(checkpoint), evaluation=str(report_path))
                save_json(folder/'status.json', state)
                print(f'FLAT_GAIT_READY={checkpoint}', flush=True)
                return
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
                 '--iterations', args.terrain_warmup_iterations, *ppo_args], 'TRAINING_FINISHED')
            checkpoint = warm/'checkpoints/final.pt'
        for cap, scale in stages:
            if cap <= state.get('validated_teacher_step_height', 0):
                continue
            levels = max(8, round((cap-.03)/.01)+1)
            label = f'{round(cap*100):02d}cm'
            best_checkpoint, best_score = checkpoint, None
            if args.resume_pipeline and state.get('training_step_height') == cap:
                previous = [json.loads(Path(path).read_text()) for path in
                            state.get('best_teacher_reports', state.get('latest_teacher_reports', []))]
                if previous and all(Path(r['checkpoint']).resolve() == Path(checkpoint).resolve() for r in previous):
                    best_score = selection_score(previous, args.gait_selection_penalty)
            for index in count():
                if args.teacher_rounds and index >= args.teacher_rounds:
                    raise RuntimeError(f'{label}教师未达标：{low_path}, {high_path}')
                teacher = run_logged(folder, f'teacher_{label}_{index}', 'scripts.train_go2_ppo',
                    ['--resume', checkpoint, '--complex', '--num-envs', 2048, '--levels', levels,
                     '--step-height-min', .03, '--step-height-max', cap, '--action-scale', scale,
                     '--height-scan-clip', 2 if cap > .125 else 1,
                     '--iterations', args.teacher_iterations,
                     '--velocity-coef', .01, '--kl-interval', 1,
                     '--seed', 42000+round(cap*100)*1000+index, *ppo_args], 'TRAINING_FINISHED')
                candidate = teacher/'checkpoints/final.pt'
                state.update(last_teacher_candidate=str(candidate), training_step_height=cap)
                save_json(folder/'status.json', state)
                low, low_path = evaluate(folder, f'teacher_{label}_low_{index}', candidate)
                high, high_path = evaluate(folder, f'teacher_{label}_high_{index}', candidate, levels-1)
                if passed(low) and passed(high):
                    checkpoint = candidate
                    state['teacher'] = str(checkpoint)
                    state['validated_teacher_step_height'] = cap
                    state.update(latest_teacher_reports=[str(low_path), str(high_path)],
                                 best_teacher_reports=[str(low_path), str(high_path)],
                                 teacher_best_score=selection_score([low, high], args.gait_selection_penalty))
                    save_json(folder/'status.json', state)
                    break
                score = selection_score([low, high], args.gait_selection_penalty)
                if best_score is None or score > best_score:
                    best_checkpoint, best_score = candidate, score
                    state['best_teacher_reports'] = [str(low_path), str(high_path)]
                checkpoint = best_checkpoint
                state.update(teacher=str(best_checkpoint), teacher_best_score=best_score,
                             teacher_round=index+1, latest_teacher_reports=[str(low_path), str(high_path)])
                save_json(folder/'status.json', state)
        if args.student_start:
            checkpoint = Path(args.student_start).resolve()
        else:
            student = run_logged(folder, 'distill', 'scripts.distill_student',
                ['--teacher', checkpoint, '--num-envs', 256,
                 '--iterations', args.distill_iterations], 'DISTILLATION_FINISHED')
            checkpoint = student/'checkpoints/final.pt'
            evaluate(folder, 'distill_eval', checkpoint)
        # 恢复时教师课程可能全部跳过，学生验收档位仍按目标地形独立计算。
        levels = max(8, round((args.max_step_height-.03)/.01)+1)
        validation_levels = sorted({0, *[round((height-.03)/(args.max_step_height-.03)*(levels-1))
                                         for height, _ in stages]})
        student_args = list(ppo_args)
        if args.student_actor_lr is not None:
            student_args += ['--actor-lr', args.student_actor_lr]
        if args.student_fixed_lr:
            student_args.append('--no-adaptive-lr')
        if args.student_freeze_proprio:
            student_args.append('--freeze-proprio')
        if args.student_action_std is not None:
            student_args += ['--target-action-std', args.student_action_std]
        if args.student_long_flight_penalty is not None:
            student_args += ['--long-flight-penalty', args.student_long_flight_penalty]
        if args.student_flight_penalty is not None:
            student_args += ['--flight-penalty', args.student_flight_penalty]
        if args.student_training_levels is not None:
            student_args += ['--training-levels', *args.student_training_levels]
        continuing = bool(args.resume_pipeline and args.student_start)
        same_student = continuing and str(checkpoint) == state.get('student')
        best_checkpoint = Path(state.get('student_best_checkpoint', checkpoint)) if same_student else checkpoint
        best_score = tuple(state['student_best_score']) if same_student and state.get('student_best_score') else None
        start_round = state.get('student_round', 0) if continuing else 0
        for index in count(start_round):
            if args.student_rounds and index >= args.student_rounds:
                raise RuntimeError(f'视觉学生全高度验收尚未达标：{path}')
            terrain_args = ([] if args.student_terrain_seed_base is None else
                            ['--terrain-seed', args.student_terrain_seed_base+index])
            student = run_logged(folder, f'student_{index}', 'scripts.train_go2_ppo',
                ['--stage', 'student_ppo', '--resume', checkpoint, '--num-envs', args.student_num_envs,
                 '--iterations', args.student_iterations, '--seed', 92000+index,
                 *student_args, *terrain_args], 'TRAINING_FINISHED')
            candidate = student/'checkpoints/final.pt'
            state['last_student_candidate'] = str(candidate)
            save_json(folder/'status.json', state)
            reports, results = [], []
            for level in validation_levels:
                report, path = evaluate(folder, f'student_level_{level}_round_{index}', candidate, level)
                reports.append(str(path))
                results.append(report)
            state['latest_student_reports'] = reports
            # 每个候选都比较全部高度，避免低档短暂退化遮住最高档的改善。
            if all(passed(report) for report in results):
                # 最终使用本轮从未参与模型选择的新seed完整复核，避免只反复挑同一组案例。
                audits = []
                for level in validation_levels:
                    report, path = evaluate(folder, f'student_audit_{level}_round_{index}', candidate,
                        level, seeds=list(range(20000+index*10, 20005+index*10)))
                    audits.append(str(path))
                    state['latest_student_audits'] = list(audits)
                    if not passed(report):
                        break
                else:
                    checkpoint = candidate
                    state.update(student=str(checkpoint), student_best_checkpoint=str(checkpoint),
                                 student_best_score=selection_score(results, args.gait_selection_penalty),
                                 best_student_reports=reports, height_evaluations=audits)
                    report_path = audits[0]
                    break
            score = selection_score(results, args.gait_selection_penalty)
            if best_score is None or score > best_score:
                best_checkpoint, best_score = candidate, score
                state['best_student_reports'] = reports
            # 续训起点和最佳候选分开，避免每个小批次都回到同一个已知有缺陷的策略。
            # 这里允许暂时退化以积累训练；passed仍使用80%/.1及全部步态硬门槛。
            keep_training = args.student_continue_routes and all(
                route_passed(r, min_success=.5, max_rmse=.2) for r in results)
            checkpoint = candidate if keep_training else best_checkpoint
            state.update(student=str(checkpoint), student_best_checkpoint=str(best_checkpoint),
                         student_best_score=best_score, student_round=index+1)
            save_json(folder/'status.json', state)
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
