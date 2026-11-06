"""确定性平地回放，量化前髋外展与机体系前足站宽，不训练或修改策略。"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import mujoco
import torch

from env.perception.runtime import VisualRuntime
from env.training_task import Go2Task
from env.posture import foot_positions_body
from env.robot import MODEL_PATH
from rl.run_io import load_policy, save_json



def home_width():
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key('home').id)
    mujoco.mj_forward(model, data)
    return float(data.geom_xpos[model.geom('FL').id, 1]-data.geom_xpos[model.geom('FR').id, 1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--steps', type=int, default=600)
    parser.add_argument('--seed', type=int, default=56100)
    args = parser.parse_args()
    torch.set_num_threads(1)
    actor, config, saved = load_policy(args.checkpoint)
    config.num_envs = 4
    config.seed = args.seed
    config.terrain.seed = args.seed
    config.randomization.seed = 200000+args.seed
    config.terrain.obstacles = False
    config.terrain.levels = 2
    config.terrain.layouts = 1
    config.terrain.width = 20.
    config.terrain.flat_length = 20.
    config.terrain.spawn_x = 10.
    config.route.enabled = False
    config.command.mode = 'legacy'
    config.command.vx = (.3, .3)
    config.command.vy = (0., 0.)
    config.command.wz = (0., 0.)
    task = Go2Task(config, curriculum=False)
    samples = []
    failed = torch.zeros(4, dtype=torch.bool, device='cuda:0')
    ids = [list(task.robot.joint_names).index(name) for name in ('FL_hip_joint', 'FR_hip_joint')]
    try:
        visual = VisualRuntime(task, actor.visual_encoder)
        for step in range(args.steps):
            with torch.no_grad():
                state, _ = task.inputs(visual.latent)
                action = actor(state)[0]
                _, terminated, truncated, _ = task.step(action)
                visual.update()
                failed |= terminated | truncated
                if step >= 50 and (~failed).any():
                    feet = foot_positions_body(task)
                    width = feet[:, 0, 1]-feet[:, 1, 1]
                    hips = task.robot.data.joint_pos[:, ids]
                    samples.append(torch.cat([width[:, None], hips], -1)[~failed].cpu())
        data = torch.cat(samples) if samples else torch.empty(0, 3)
        report = dict(checkpoint=str(Path(args.checkpoint).resolve()), iteration=saved['iteration'],
            audit='12秒平地定速0.3，不是完整路线验收', env_config=asdict(config),
            joint_order=list(task.robot.joint_names), home_front_width_m=home_width(),
            samples=len(data), failed_environments=int(failed.sum()),
            front_width_mean_m=data[:, 0].mean().item() if len(data) else None,
            front_width_p95_m=data[:, 0].quantile(.95).item() if len(data) else None,
            front_width_max_m=data[:, 0].max().item() if len(data) else None,
            front_hip_mean_rad=data[:, 1:].mean(0).tolist() if len(data) else None,
            front_hip_abs_p95_rad=data[:, 1:].abs().quantile(.95, dim=0).tolist() if len(data) else None)
        save_json(args.output, report)
        print(json.dumps({k: v for k, v in report.items() if k!='env_config'}, ensure_ascii=False), flush=True)
    finally:
        task.close()


if __name__ == '__main__':
    main()
