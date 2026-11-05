"""固定已验证的共享控制网络，用多个真实地形seed连续细化视觉分支。"""

import argparse
from datetime import datetime
import json
from pathlib import Path

from rl.run_io import ROOT, save_json
from scripts.train_pipeline import run_logged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--teacher', required=True)
    parser.add_argument('--student', required=True)
    parser.add_argument('--terrain-seeds', type=int, nargs='+', required=True)
    parser.add_argument('--iterations', type=int, default=50, help='每个地形seed的新增更新次数')
    parser.add_argument('--num-envs', type=int, default=256)
    parser.add_argument('--visual-lr', type=float, default=3e-5)
    parser.add_argument('--teacher-mix', type=float, default=.5)
    parser.add_argument('--latent-coef', type=float, default=.1)
    parser.add_argument('--copy-teacher-common', action='store_true', help='首阶段恢复教师共享控制器')
    parser.add_argument('--action-noise-std', type=float, default=0.)
    parser.add_argument('--training-levels', type=int, nargs='+',
                        default=[0, 6, 9, 12, 15, 17, 17, 17, 17, 17])
    args = parser.parse_args()
    teacher, checkpoint = Path(args.teacher).resolve(), Path(args.student).resolve()
    folder = ROOT/'outputs'/f'{datetime.now():%Y%m%d_%H%M%S}_visual_refinement'
    folder.mkdir()
    save_json(folder/'arguments.json', vars(args))
    state = dict(folder=str(folder), teacher=str(teacher), source_student=str(checkpoint),
                 control_reference=str(teacher if args.copy_teacher_common else checkpoint),
                 status='running', results=[])
    save_json(folder/'status.json', state)
    print(f'REFINEMENT_DIR={folder}', flush=True)
    try:
        for index, seed in enumerate(args.terrain_seeds):
            name = f'visual_terrain_{seed}'
            state.update(stage=name, log=str(folder/f'{name}.log'))
            save_json(folder/'status.json', state)
            run = run_logged(folder, name, 'scripts.distill_student',
                ['--teacher', teacher, '--resume', checkpoint, '--num-envs', args.num_envs,
                 '--iterations', args.iterations, '--seed', seed+1000, '--terrain-seed', seed,
                 '--visual-lr', args.visual_lr, '--common-lr', 0, '--freeze-common', '--warmup', 0,
                 '--teacher-mix', args.teacher_mix, '--latent-coef', args.latent_coef,
                 '--action-noise-std', args.action_noise_std,
                 *(['--copy-teacher-common'] if args.copy_teacher_common and index == 0 else []),
                 '--training-levels', *args.training_levels],
                'DISTILLATION_FINISHED')
            metadata = json.loads((run/'metadata.json').read_text())
            geometry = metadata['terrain_geometry_sha256']
            if any(row['geometry_sha256'] == geometry for row in state['results']):
                raise RuntimeError('地形seed改变但编译几何重复，停止本轮视觉细化')
            checkpoint = run/'checkpoints/final.pt'
            state['results'].append(dict(terrain_seed=seed, run=str(run),
                                          checkpoint=str(checkpoint), geometry_sha256=geometry))
            state.update(checkpoint=str(checkpoint), additional_visual_updates=len(state['results'])*args.iterations)
            save_json(folder/'status.json', state)
        state['status'] = 'finished'
        save_json(folder/'status.json', state)
        print(f'VISUAL_REFINEMENT_FINISHED={checkpoint}', flush=True)
    except BaseException as error:
        state.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error=str(error))
        save_json(folder/'status.json', state)
        raise


if __name__ == '__main__':
    main()
