"""用真值地形特征隔离共享控制器误差；输出仅供诊断，不属于视觉策略验收。"""

import argparse
import copy
from datetime import datetime
from pathlib import Path

import torch

from env.gait import summarize_gait, gait_passed
from env.training_task import Go2Task
from rl.run_io import ROOT, load_policy, save_json


@torch.no_grad()
def probe(student, teacher, config, level, control):
    task = Go2Task(config, curriculum=False)
    try:
        terrain = task.env.scene.terrain
        terrain.terrain_levels[:] = level
        terrain.env_origins[:] = terrain.terrain_origins[level, terrain.terrain_types]
        task.reset()
        complete = torch.zeros(config.num_envs, dtype=torch.bool, device=task.device)
        longest = torch.zeros(config.num_envs, device=task.device)
        flight_x = torch.zeros_like(longest)
        records = []
        for step in range(task.env.max_episode_length):
            privileged, _ = task.inputs()
            if control == 'teacher':
                action = teacher(privileged)[0]
            else:
                z = teacher.terrain_encoder(privileged[:, 495:])
                action = student(torch.cat([privileged[:, :495], z], -1))[0]
            action[complete] = 0
            _, terminated, truncated, info = task.step(action)
            duration = task.gait.total[:, 2]
            local_x = (task.robot.data.root_link_pos_w[:, 0]-task.env.scene.env_origins[:, 0]
                       +config.terrain.spawn_x)
            flight_x = torch.where(duration > longest, local_x, flight_x)
            longest = torch.maximum(longest, duration)
            done = terminated | truncated
            for i in (done & ~complete).nonzero(as_tuple=False).flatten().tolist():
                gait = task.gait.snapshot(i)
                gait['all_max_flight_local_x'] = flight_x[i].item() if longest[i].item() > 0 else None
                records.append(dict(layout=int(terrain.terrain_types[i]), success=bool(info['success'][i]),
                                    seconds=(step+1)*task.dt, vx_rmse=info['vx_rmse'][i].item(), gait=gait))
            complete |= done
            if complete.all():
                break
            task.reset(done.nonzero(as_tuple=False).flatten())
        gait = summarize_gait(records)
        return dict(control=control, level=level, stair_height=config.terrain.step_height(level/(config.terrain.levels-1)),
                    cases=len(records), route_success_rate=sum(r['success'] for r in records)/len(records),
                    vx_rmse=sum(r['vx_rmse'] for r in records)/len(records),
                    gait_probe=gait_passed(dict(gait=gait)), gait=gait, records=records)
    finally:
        task.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--student', required=True)
    parser.add_argument('--teacher', required=True)
    parser.add_argument('--levels', type=int, nargs='+', default=[6, 12, 17])
    parser.add_argument('--controls', nargs='+', choices=['teacher', 'student_oracle'],
                        default=['teacher', 'student_oracle'])
    parser.add_argument('--num-envs', type=int, default=4)
    parser.add_argument('--seed', type=int, default=10000)
    args = parser.parse_args()
    torch.set_num_threads(1)
    student, template, _ = load_policy(args.student)
    teacher, _, _ = load_policy(args.teacher)
    folder = ROOT/'outputs'/f'{datetime.now():%Y%m%d_%H%M%S}_control_oracle_diagnostic'
    folder.mkdir()
    report = dict(purpose='ground-truth feature isolation; not visual policy acceptance',
                  uses_ground_truth_terrain=True, student=str(Path(args.student).resolve()),
                  teacher=str(Path(args.teacher).resolve()), terrain_seed=args.seed,
                  status='running', results=[])
    save_json(folder/'diagnostic.json', report)
    print(f'DIAGNOSTIC_DIR={folder}', flush=True)
    try:
        for level in args.levels:
            for control in args.controls:
                config = copy.deepcopy(template)
                config.num_envs, config.seed, config.terrain.seed = args.num_envs, args.seed+1000, args.seed
                result = probe(student, teacher, config, level, control)
                report['results'].append(result)
                save_json(folder/'diagnostic.json', report)
                print({k:result[k] for k in ['control', 'stair_height', 'route_success_rate', 'gait_probe']}, flush=True)
        report['status'] = 'finished'
        save_json(folder/'diagnostic.json', report)
    except BaseException as error:
        report.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error=str(error))
        save_json(folder/'diagnostic.json', report)
        raise


if __name__ == '__main__':
    main()
