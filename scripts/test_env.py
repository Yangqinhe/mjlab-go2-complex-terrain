"""独立验证 Go2 与混合地形同场加载、固定指令和 GPU 步进，可导出场景图。"""

import argparse
import copy
from dataclasses import asdict
from datetime import datetime
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import time
from zoneinfo import ZoneInfo

import mujoco
import numpy as np
import torch

from env.config import EnvConfig
from env.go2env import create_env
from env.robot import MODEL_PATH

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def inspect_terrain(env, config):
    """直接对最终编译碰撞模型射线检查，避免只检查配置里是否写了地形名称。"""
    model = env.sim.mj_model
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    terrain = env.scene.terrain
    course = config.terrain
    origins = terrain.terrain_origins.cpu().numpy()
    rays = []
    group = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
    for level in range(course.levels):
        height = course.step_height(level / (course.levels - 1))
        origin = origins[level, 0]
        corner = origin - np.array([course.spawn_x, course.width / 2, 0.0])
        samples = []
        for name, (start, end) in course.regions().items():
            if name in ("stairs_up", "stairs_down"):
                for index in range(course.step_count):
                    step = index + 1 if name == "stairs_up" else course.step_count-index-1
                    samples.append((f"{name}_{index}", start+(index+0.5)*course.step_depth,
                                    step*height))
            else:
                expected = course.step_count*height if name == "platform" else 0.0
                samples.append((name, (start+end)/2, None if name == "rubble" else expected))
        for name, x, expected in samples:
            point = corner + np.array([x, course.width / 2, 2.0])
            geom_id = np.array([-1], dtype=np.int32)
            distance = mujoco.mj_ray(model, data, point, np.array([0., 0., -1.]),
                                     group, 1, -1, geom_id)
            assert distance >= 0, f"Missing ground: {level=} {name=}"
            top = float(point[2] - distance)
            if expected is not None:
                assert abs(top-expected) < 1e-5, (level, name, top, expected)
            rays.append({"level": level, "region": name, "height_m": top,
                         "geom": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM,
                                                  int(geom_id[0]))})
    geometry = []
    for index in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, index) or ""
        if name.startswith("course_"):
            geometry.append({"name": name, "world_position": data.geom_xpos[index].tolist(),
                             "half_size": model.geom_size[index].tolist()})
    assert len(geometry) == course.levels * course.layouts * (
        1 + 2 * course.step_count - 1 + 1 + course.rubble_rows*course.rubble_columns
    ), "Unexpected mixed terrain geometry count"
    return {"config": asdict(course), "origins": origins.tolist(), "rays": rays,
            "geometry": geometry}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--levels", type=int, default=3)
    parser.add_argument("--layouts", type=int, default=1)
    parser.add_argument("--step-height-max", type=float, default=.09)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--level", type=int, default=0, help="初始最高地形等级（从 0 开始）")
    parser.add_argument("--render", action="store_true", help="离屏导出总览和机器人近景 PNG")
    parser.add_argument("--viewer", action="store_true", help="完成检查后启动交互窗口")
    args = parser.parse_args()
    if args.num_envs < 1 or args.steps < 1:
        parser.error("num-envs 和 steps 必须为正整数")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用；CPU 验证请显式指定 --device cpu")
    config = EnvConfig(num_envs=args.num_envs)
    config.terrain.initial_level = args.level
    config.terrain.levels, config.terrain.layouts = args.levels, args.layouts
    config.terrain.step_height_max = args.step_height_max
    stamp = datetime.now(ZoneInfo("Asia/Hong_Kong")).strftime("%Y%m%d_%H%M%S_%f")
    output = PROJECT_ROOT / "outputs" / f"{stamp}_load_env_seed{config.seed}"
    output.mkdir(parents=True)
    save_json(output/"config.json", asdict(config))
    save_json(output/"randomization.json", {"enabled": False})
    print(f"Output: {output}", flush=True)
    start = time.perf_counter()
    env = create_env(config, args.device)
    try:
        obs, _ = env.reset()
        robot = env.scene["robot"]
        assert tuple(obs["actor"].shape) == (args.num_envs, 48)
        assert tuple(env.action_space.shape) == (args.num_envs, 12)
        assert env.sim.mj_model.nu == 12, "Duplicated or missing joint actuators"
        expected_joints = [f"{leg}_{joint}_joint" for leg in ["FL", "FR", "RL", "RR"]
                           for joint in ["hip", "thigh", "calf"]]
        assert list(robot.joint_names) == expected_joints
        manifest = inspect_terrain(env, config)
        save_json(output/"terrain_manifest.json", manifest)
        initialized = time.perf_counter()-start
        actions = torch.zeros((args.num_envs, 12), device=args.device)
        expected_command = torch.tensor([0.3, 0., 0.], device=args.device)
        reset_count = 0
        minimum_height = float("inf")
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            for _ in range(args.steps):
                # 零动作只验证默认姿态 PD，不对机器人施加 vx=0.3 的速度。
                obs, reward, terminated, truncated, _ = env.step(actions)
                assert torch.isfinite(obs["actor"]).all()
                assert torch.isfinite(reward).all()
                command = env.command_manager.get_command("velocity")
                assert torch.allclose(command, expected_command.expand_as(command))
                assert torch.allclose(obs["actor"][:, 9:12], command)
                assert torch.isfinite(robot.data.joint_pos).all()
                minimum_height = min(minimum_height,
                                     robot.data.root_link_pos_w[:, 2].min().item())
                done = (terminated | truncated).nonzero(as_tuple=False).flatten()
                if len(done):
                    reset_count += len(done)
                    env.reset(env_ids=done)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = time.perf_counter()-start
        # 用最终 GPU 状态检查一次 CPU 接触，确认碰撞几何不只是可见装饰。
        model = env.sim.mj_model
        data = mujoco.MjData(model)
        data.qpos[:] = env.sim.data.qpos[0].cpu().numpy()
        data.qvel[:] = env.sim.data.qvel[0].cpu().numpy()
        mujoco.mj_forward(model, data)
        terrain_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "terrain")
        terrain_contacts = sum(
            terrain_body in [model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]]
            for c in data.contact
        )
        assert terrain_contacts > 0, "Final robot state has no terrain contact"
        warp_contacts = int(env.sim.data.nacon[0].item())
        assert warp_contacts > 0, "MuJoCo Warp detected no contacts"
        report = {
            "status": "passed", "num_envs": args.num_envs, "steps_per_env": args.steps,
            "device": args.device, "gpu": torch.cuda.get_device_name(args.device)
            if args.device.startswith("cuda") else "cpu",
            "command_m_s_rad_s": [0.3, 0.0, 0.0],
            "observation_shape": list(obs["actor"].shape), "action_shape": list(actions.shape),
            "joint_order": list(robot.joint_names), "terrain_contact_count_env0": terrain_contacts,
            "warp_detected_contacts_all_worlds": warp_contacts,
            "initial_terrain_levels": env.scene.terrain.terrain_levels.cpu().tolist(),
            "terrain_collision_geometries": len(manifest["geometry"]),
            "validated_terrain_rays": len(manifest["rays"]),
            "step_heights_m": [config.terrain.step_height(i/(config.terrain.levels-1))
                               for i in range(config.terrain.levels)],
            "initialization_seconds": initialized, "step_seconds": elapsed,
            "environment_steps_per_second": args.num_envs*args.steps/elapsed,
            "resets": reset_count, "minimum_base_height_m": minimum_height,
            "actual_mean_vx_m_s": float(robot.data.root_link_lin_vel_b[:, 0].mean()),
            "model_xml_sha256": hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),
            "versions": {p: metadata.version(p) for p in ["mjlab", "mujoco", "mujoco-warp",
                                                        "warp-lang", "torch"]},
            "note": "加载与 PD 零动作检查；未训练策略，不能证明 0.3 m/s 行走或跨越地形。",
        }
        save_json(output/"report.json", report)
        if args.render:
            save_previews(env, config, data, output)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        if args.viewer:
            from mjlab.viewer import NativeMujocoViewer
            # 交互 viewer 不负责手动 reset；只在展示阶段启用自动 reset。
            env.cfg.auto_reset = True
            NativeMujocoViewer(env, lambda _: actions).run()
    finally:
        env.close()


def save_previews(env, config, data, output):
    """使用实际最终状态离屏渲染；不是训练视频或经过训练的策略表现。"""
    import imageio.v3 as iio
    origin = env.scene.env_origins[0].cpu().numpy()
    model = copy.copy(env.sim.mj_model)
    level = int(env.scene.terrain.terrain_levels[0])
    active_prefix = f"course_{round(level/(config.terrain.levels-1)*1000):04d}"
    if config.terrain.layouts > 1:
        active_prefix += f"_v{int(env.scene.terrain.terrain_types[0])}"
    active_prefix += "_"
    # 仅在渲染副本中隐藏其他难度路线，防止画面被相邻地块干扰。
    for index in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, index) or ""
        if name.startswith("course_") and not name.startswith(active_prefix):
            model.geom_group[index] = 5
    option = mujoco.MjvOption()
    option.geomgroup[:] = [1, 1, 1, 0, 0, 0]
    with mujoco.Renderer(model, height=720, width=1280) as renderer:
        for name, distance, lookat in [
            ("scene_overview", 18.0, origin + np.array([
                config.terrain.length/2-config.terrain.spawn_x, 0., 0.])),
            ("robot_closeup", 2.0, data.qpos[:3]),
        ]:
            camera = mujoco.MjvCamera()
            camera.lookat[:] = lookat
            camera.distance = distance
            camera.azimuth = 115
            camera.elevation = -38
            renderer.update_scene(data, camera=camera, scene_option=option)
            iio.imwrite(output/f"{name}.png", renderer.render())


if __name__ == "__main__":
    main()
