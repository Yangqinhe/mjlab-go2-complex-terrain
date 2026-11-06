"""集中定义续训扰动范围；默认关闭，旧权重的环境行为不变。"""

from dataclasses import dataclass


@dataclass
class RandomizationConfig:
    enabled: bool = False
    seed: int = 81000
    nominal_fraction: float = .2
    payload_max: float = .5
    mass_scale: tuple[float, float] = (.95, 1.05)
    com_error: float = .003
    friction: tuple[float, float] = (.65, 1.0)
    gain_scale: tuple[float, float] = (.95, 1.05)
    effort_scale: tuple[float, float] = (.95, 1.)
    damping_scale: tuple[float, float] = (.9, 1.1)
    gyro_noise: float = .01
    gyro_bias: float = .005
    push_force: float = 0.  # 后续课程才开启小外力，首批0表示关闭。
    position_noise: float = .003
    velocity_noise: float = .05
    joint_bias: float = .005
    gravity_angle: float = .008726646
    action_delay_steps: int = 1  # 50Hz整数周期，实际为0或20ms，不冒称任意毫秒延迟。
    camera_pitch_degrees: tuple[float, float] | None = None
    camera_position: float = .003
    camera_angle: float = .008726646
    depth_scale_error: float = 0.
    depth_patch_probability: float = 0.
    depth_noise: float = .002
    depth_dropout: float = .01
    frame_dropout: float = .01
    frame_delay_steps: int = 1  # 50Hz控制周期；只在新帧到达时更新GRU。


def d435i_scene_config(source):
    from copy import deepcopy
    from env.perception.config import DepthConfig
    config=deepcopy(source)
    config.num_envs=64;config.seed=82100;config.terrain.seed=72100
    config.command.mode='legacy';config.command.vx=(.3,.3);config.command.vy=(0.,0.);config.command.wz=(0.,0.)
    config.route.enabled=True;config.episode_seconds=90.
    config.terrain.obstacles=True;config.terrain.levels=18
    config.terrain.width=3.;config.terrain.flat_length=3.;config.terrain.spawn_x=1.
    config.terrain.surroundings=True;config.terrain.layouts=4
    config.randomization.enabled=True
    config.randomization.seed=82100
    config.camera=DepthConfig(fovx_degrees=87.,fovy_degrees=58.,pitch_down_degrees=35.)
    config.randomization.camera_pitch_degrees=(20.,50.)
    config.randomization.camera_position=.01
    config.randomization.depth_scale_error=.01
    config.randomization.depth_patch_probability=.15
    return config
