"""核对最终模型高度、光轴深度、176维网络、10Hz采样及局部reset隔离。"""

from dataclasses import asdict
import json
import math

import imageio.v3 as iio
import mujoco
import numpy as np
import torch

from env.config import EnvConfig
from env.training_task import Go2Task
from env.perception.runtime import VisualRuntime
from rl.algorithms.perceptive_networks import PerceptiveActor, PrivilegedCritic
from rl.config import TrainConfig
from rl.run_io import new_run, save_json


def main():
    torch.set_num_threads(1)
    config = EnvConfig(num_envs=4)
    task = Go2Task(config)
    actor = PerceptiveActor('student').cuda()
    visual = VisualRuntime(task, actor.visual_encoder)
    folder = new_run('perception_check', config, TrainConfig())
    model = task.env.sim.mj_model
    data = mujoco.MjData(model)
    data.qpos[:] = task.env.sim.data.qpos[0].cpu().numpy()
    mujoco.mj_forward(model, data)
    group = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
    error = []
    # 核对角落/中心射线：直接用相机实际世界方向，Z-depth需乘同一光轴因子。
    for ray in [0, 63, 24*64+32, 47*64, 48*64-1]:
        point = visual.sensor.pnt[0, ray].cpu().numpy().astype(float)
        direction = visual.sensor.vec[0, ray].cpu().numpy().astype(float)
        gid = np.array([-1], dtype=np.int32)
        distance = mujoco.mj_ray(model, data, point, direction, group, 1, -1, gid)
        expected = distance*float(-visual.sensor.directions[ray, 2]) if distance >= 0 else 3.
        actual = visual.sensor.raw_depth.flatten(1)[0, ray].item()
        error.append(abs(expected-actual))
    assert max(error) < .002, error
    xy = task.env.scene.env_origins[:, None, :2].expand(-1, 3, -1).clone()
    xy[:, :, 0] += torch.tensor([0., 2.175, 3.775], device=task.device)
    height = task.ground(xy)
    assert torch.allclose(height[:, 0], torch.zeros(4, device=task.device), atol=1e-5)
    assert torch.allclose(height[:, 1], torch.full((4,), .03, device=task.device), atol=1e-5)
    initial = visual.sensor.capture_count
    for _ in range(10):
        with torch.no_grad():
            states, critic = task.inputs(visual.latent)
            mu, std = actor(states)
            assert mu.shape == (4, 12) and critic.shape == (4, 244)
            assert actor.actor[0].in_features == 176
            reward, terminated, truncated, _ = task.step(torch.zeros_like(mu))
            assert torch.isfinite(reward).all()
            visual.update()
    assert visual.sensor.capture_count-initial == 2
    hidden = visual.hidden.clone()
    task.reset(torch.tensor([0], device=task.device))
    visual.reset(torch.tensor([0], device=task.device))
    assert torch.equal(hidden[1:], visual.hidden[1:]), 'Reset leaked to other worlds'
    # 仅为相机标定把机器人放到台阶前1m；这不是策略穿越成功的证据。
    root = task.robot.data.default_root_state.clone()
    root[:, :3] += task.env.scene.env_origins
    root[:, 0] += 1.0
    task.robot.write_root_state_to_sim(root)
    task.env.sim.forward()
    visual.sensor.capture()
    direction = visual.sensor.vec[0]
    flat_range = (-visual.sensor.pnt[0, :, 2]/direction[:, 2].clamp(max=-1e-6))
    flat_depth = flat_range*(-visual.sensor.directions[:, 2])
    flat_depth = flat_depth.clamp(.15, 3)
    flat_depth = torch.where(direction[:, 2]<0, flat_depth, 3.)
    actual = visual.sensor.raw_depth[0].flatten().clamp(.15, 3)
    visible_steps = int(((actual-flat_depth).abs()>.005).sum())
    assert visible_steps > 3, 'Small stair has no visible depth signature'
    np.save(folder/'depth_meters.npy', visual.sensor.raw_depth[0].cpu().numpy())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(9, 4))
    shown = axes[0].imshow(visual.sensor.raw_depth[0].cpu(), vmin=.15, vmax=3, cmap='viridis')
    figure.colorbar(shown, ax=axes[0], label='Z-depth (m)')
    axes[0].set_title('3 cm stairs, robot 1 m before first step')
    axes[1].plot(actual.view(48, 64)[:, 32].cpu(), label='mixed terrain')
    axes[1].plot(flat_depth.view(48, 64)[:, 32].cpu(), '--', label='flat reference')
    axes[1].set(xlabel='image row', ylabel='Z-depth (m)')
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(folder/'depth_calibration.png', dpi=160)
    plt.close(figure)
    image = ((visual.sensor.raw_depth[0].clamp(.15, 3)-.15)/2.85*255).byte().cpu().numpy()
    iio.imwrite(folder/'depth.png', image)
    report = dict(status='passed', max_cpu_depth_error_m=max(error),
                  depth_shape=list(visual.pair.shape), actor_fusion_dim=176,
                  normal_captures=visual.normal_captures, reset_captures=visual.reset_captures,
                  height_query_m=height.cpu().tolist(), reset_isolation=True,
                  small_stair_visible_pixels=visible_steps,
                  actor_parameters=sum(p.numel() for p in actor.parameters()))
    save_json(folder/'report.json', report)
    save_json(folder/'perception.json', asdict(visual.sensor.config))
    print(json.dumps(report, indent=2), flush=True)
    print(f'OUTPUT={folder}', flush=True)
    task.close()


if __name__ == '__main__':
    main()
