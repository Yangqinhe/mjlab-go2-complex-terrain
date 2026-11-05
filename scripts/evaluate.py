"""用独立地形seed评估确定性教师/学生，可离屏记录RGB与深度视频。"""

import argparse
import copy
import csv
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch

from env.training_task import Go2Task
from env.perception.runtime import VisualRuntime
from env.gait import summarize_gait
from rl.run_io import load_policy, save_json


def evaluate(checkpoint, num_envs=4, seeds=(10000,), level=0, device='cuda:0',
             video=None, max_steps=None, route_following=None):
    actor, original, saved = load_policy(checkpoint, device)
    if route_following is not None:
        original.route.enabled = route_following
    records = []
    gait_trace = []
    writer = imageio.get_writer(str(video), fps=30, codec='libx264', quality=7) if video else None
    try:
        for terrain_seed in seeds:
            config = copy.deepcopy(original)
            config.num_envs, config.seed = num_envs, 1000+terrain_seed
            config.terrain.seed = terrain_seed
            task = Go2Task(config, device, curriculum=False)
            try:
                terrain = task.env.scene.terrain
                terrain.terrain_levels[:] = level
                terrain.env_origins[:] = terrain.terrain_origins[level, terrain.terrain_types]
                task.reset()
                visual = VisualRuntime(task, actor.visual_encoder) if actor.mode == 'student' else None
                complete = torch.zeros(num_envs, dtype=torch.bool, device=device)
                renderer = None
                if writer:
                    model = copy.copy(task.env.sim.mj_model)
                    # MuJoCo裁剪距离按整张地图extent缩放；大课程地图会裁掉2.5m跟拍中的机身。
                    model.vis.map.znear = .02/model.stat.extent
                    model.vis.map.zfar = 50./model.stat.extent
                    active = f'course_{round(level/(config.terrain.levels-1)*1000):04d}'
                    if config.terrain.layouts > 1:
                        active += f'_v{int(terrain.terrain_types[0])}'
                    for i in range(model.ngeom):
                        if model.geom(i).name.startswith('course_') and not model.geom(i).name.startswith(active+'_'):
                            model.geom_group[i] = 5
                    data = mujoco.MjData(model)
                    renderer = mujoco.Renderer(model, height=720, width=1280)
                    option = mujoco.MjvOption()
                    option.geomgroup[:] = [1, 1, 1, 0, 0, 0]
                    camera = mujoco.MjvCamera()
                    camera.distance, camera.elevation, camera.azimuth = 2.5, -25, 110
                next_frame = 0.
                limit = max_steps or task.env.max_episode_length
                for step in range(limit):
                    # task持有跨步缓存，后续reset需要原地写入，不能创建inference tensor。
                    with torch.no_grad():
                        states, _ = task.inputs(None if visual is None else visual.latent)
                        action = actor(states)[0]
                        action[complete] = 0
                        reward, terminated, truncated, info = task.step(action)
                        if visual:
                            visual.update()
                    if writer:
                        contacts = task.gait.previous_contact[0].cpu().tolist()
                        gait_trace.append(dict(seed=terrain_seed, seconds=(step+1)*task.dt,
                            **dict(zip(('FL', 'FR', 'RL', 'RR'), map(int, contacts))),
                            height=info['height'][0].item(), vx=info['velocity'][0].item()))
                    if writer and step*task.dt+1e-9 >= next_frame:
                        data.qpos[:] = task.env.sim.data.qpos[0].cpu().numpy()
                        data.qvel[:] = task.env.sim.data.qvel[0].cpu().numpy()
                        mujoco.mj_forward(model, data)
                        camera.lookat[:] = data.qpos[:3]
                        renderer.update_scene(data, camera=camera, scene_option=option)
                        frame = renderer.render().copy()
                        if visual:
                            depth = (visual.sensor.raw_depth[0].clamp(.15, 3)/3*255).byte().cpu().numpy()
                            depth = np.repeat(np.repeat(depth, 15, 0), 15, 1)
                            frame = np.concatenate([frame, np.repeat(depth[:, :, None], 3, 2)], 1)
                        from PIL import Image, ImageDraw, ImageFont
                        image = Image.fromarray(frame)
                        draw = ImageDraw.Draw(image)
                        draw.rectangle((0, 0, frame.shape[1], 48), fill=(0, 0, 0))
                        draw.text((15, 10),
                            f"{actor.mode} | target vx=0.30 | actual vx={info['velocity'][0].item():.3f} | "
                            f"time={step*task.dt:.1f}s | level={level} | seed={terrain_seed}",
                            fill=(255, 255, 255), font=ImageFont.load_default(size=24))
                        draw.text((15, 35), 'contact: '+ '  '.join(
                            f'{leg}={int(on)}' for leg, on in zip(('FL', 'FR', 'RL', 'RR'), contacts)),
                            fill=(255, 255, 255), font=ImageFont.load_default(size=12))
                        writer.append_data(np.asarray(image))
                        next_frame += 1/30
                    done = terminated | truncated
                    if step == limit-1:
                        done = torch.ones_like(done)
                    new = (done & ~complete).nonzero(as_tuple=False).flatten()
                    for i in new.tolist():
                        records.append(dict(seed=terrain_seed, layout=int(terrain.terrain_types[i]),
                            level=level, success=bool(info['success'][i]), fall=bool(info['fall'][i]),
                            outside=bool(info['outside'][i]), timeout=bool(truncated[i]),
                            numerical_failure=bool(info['numerical_failure'][i]),
                            seconds=(step+1)*task.dt, distance=info['progress'][i].item(),
                            rmse=info['vx_rmse'][i].item(),
                            planar_rmse=info['rmse'][i].item(), reward=info['episode_return'][i].item(),
                            evaluation_budget_end=bool(step == limit-1 and not (terminated[i] | truncated[i])),
                            regions=task.visited[i].cpu().tolist()))
                        records[-1]['gait'] = task.gait.snapshot(i)
                    complete |= done
                    if complete.all():
                        break
                    ids = done.nonzero(as_tuple=False).flatten()
                    if len(ids):
                        task.reset(ids)
                        if visual:
                            visual.reset(ids)
                if renderer:
                    renderer.close()
            finally:
                task.close()
    finally:
        if writer:
            writer.close()
            if gait_trace:
                with Path(video).with_suffix('.gait.csv').open('w', newline='') as stream:
                    output = csv.DictWriter(stream, fieldnames=list(gait_trace[0]))
                    output.writeheader()
                    output.writerows(gait_trace)
    region_names = list(original.terrain.regions())
    report = dict(checkpoint=str(Path(checkpoint).resolve()), stage=saved['stage'],
                  iteration=saved['iteration'], episodes=len(records),
                  unique_layouts=len({(r['seed'], r['layout'], r['level']) for r in records}),
                  level=level, seeds=list(seeds),
                  route_following=original.route.enabled,
                  route_override=route_following,
                  terrain_obstacles=original.terrain.obstacles,
                  stair_height=original.terrain.step_height(level/max(original.terrain.levels-1, 1))
                               if original.terrain.obstacles else 0.,
                  evaluation_step_limit=max_steps,
                  partial_evaluation=max_steps is not None and max_steps < task.env.max_episode_length,
                  success_rate=float(np.mean([r['success'] for r in records])),
                  fall_rate=float(np.mean([r['fall'] for r in records])),
                  outside_rate=float(np.mean([r['outside'] for r in records])),
                  timeout_rate=float(np.mean([r['timeout'] for r in records])),
                  numerical_failure_rate=float(np.mean([r['numerical_failure'] for r in records])),
                  walking_gait=original.gait.enabled, gait=summarize_gait(records, task.dt),
                  vx_rmse=float(np.mean([r['rmse'] for r in records])),
                  mean_distance=float(np.mean([r['distance'] for r in records])),
                  mean_seconds=float(np.mean([r['seconds'] for r in records])),
                  region_reach={name:float(np.mean([r['regions'][i] for r in records]))
                                for i, name in enumerate(region_names)}, records=records)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--num-envs', type=int, default=4)
    parser.add_argument('--seeds', type=int, nargs='+', default=[10000])
    parser.add_argument('--level', type=int, default=0)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--video')
    parser.add_argument('--max-steps', type=int)
    parser.add_argument('--route-following', action='store_true', default=None)
    args = parser.parse_args()
    torch.set_num_threads(1)
    report = evaluate(args.checkpoint, args.num_envs, args.seeds, args.level,
                      args.device, args.video, args.max_steps, args.route_following)
    stamp = datetime.now(ZoneInfo('Asia/Hong_Kong')).strftime('%Y%m%d_%H%M%S_%f')
    folder = Path(args.checkpoint).resolve().parent.parent/'evaluations'
    folder.mkdir(exist_ok=True)
    path = folder/f'eval_{stamp}.json'
    save_json(path, report)
    print(json.dumps({k:v for k,v in report.items() if k != 'records'}), flush=True)
    print(f'EVALUATION={path}', flush=True)


if __name__ == '__main__':
    main()
