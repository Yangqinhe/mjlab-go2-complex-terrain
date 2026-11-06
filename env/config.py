"""集中管理 Go2 控制、固定速度指令和环境参数。"""

from dataclasses import dataclass, field

from terrains.config import TerrainConfig
from domain_randomization.config import RandomizationConfig
from env.perception.config import DepthConfig


@dataclass
class VelocityCommandConfig:
    # 预留范围采样；当前上下界相等，因此 reset 和回合内都保持固定指令。
    vx: tuple[float, float] = (0.3, 0.3)
    vy: tuple[float, float] = (0.0, 0.0)
    wz: tuple[float, float] = (0.0, 0.0)
    resample_seconds: float = 3.0
    mode: str = "legacy"
    yaw_limit: float = .5
    acceleration: float = .2
    yaw_acceleration: float = .5
    stop_fraction: float = .15
    anchor_fraction: float = .2
    stop_script_fraction: float = 0.
    stop_hold_seconds: tuple[float, float] | None = None


@dataclass
class RouteCommandConfig:
    enabled: bool = False
    lateral_gain: float = .7
    heading_gain: float = 2.
    max_yaw_rate: float = .5


@dataclass
class GaitConfig:
    enabled: bool = False
    nominal_height: float = .27
    contact_force: float = 5.
    flat_margin: float = .35
    minimum_swing: float = .08
    flight: float = -2.
    # 后期可单独提高持续腾空成本；默认0保留旧checkpoint的奖励。
    long_flight: float = 0.
    flight_grace: float = .04
    flat_flight: float = -2.
    low_support: float = -2.
    # 可选的复杂地形落足重叠：2足中性，3/4足奖励，0/1足惩罚；旧配置保持0。
    terrain_support: float = 0.
    terrain_support_downstairs_only: bool = False
    wrong_pair: float = -2.
    diagonal_support: float = .4
    vertical: float = -3.
    height: float = -5.
    touchdown: float = 5.
    short_swing: float = -5.


@dataclass
class TrainingRewardConfig:
    # 后续前腿微调专用；原任务和历史checkpoint默认关闭。
    all_hip_deviation: float = 0.
    foot_landing: float = 0.
    foot_home_x: tuple[float, ...] = (.1934, .1934, -.1934, -.1934)
    foot_home_y: tuple[float, ...] = (.142, -.142, .142, -.142)
    foot_landing_x_tolerance: float = .16
    foot_landing_y_tolerance: float = .04
    front_hip_deviation: float = 0.
    front_stance_width: float = 0.
    front_hip_deadband: float = .15
    front_width_target: float = .284
    front_width_tolerance: float = .04
    tracking_xy: float = 1.5
    tracking_yaw: float = .5
    tracking_yaw_scale: float = .25
    standing_velocity: float = 5.
    standing_joint_velocity: float = .02
    standing_angular: float = 0.
    progress: float = .2
    orientation: float = -2.
    angular_xy: float = -.05
    vertical: float = -.5
    height: float = -2.
    effort: float = -.00002
    action_rate: float = -.01
    joint_limit: float = -1.
    slip: float = -.2
    collision: float = -.5
    foot_clearance: float = -5.
    route_lateral: float = -1.
    route_yaw: float = -.8
    swing_clearance: float = .03
    lookahead_distance: float = .20
    success: float = 5.
    failure: float = -1.
    nominal_height: float = .27
    minimum_height: float = .13


@dataclass
class EnvConfig:
    num_envs: int = 1
    seed: int = 42
    physics_dt: float = 0.002
    decimation: int = 10
    episode_seconds: float = 90.0
    kp: float = 40.0
    kd: float = 2.0
    action_scale: float = 0.5
    height_scan_clip: float = 1.0
    base_height: float = 0.30
    tracking_weight: float = 1.5
    tracking_error_scale: float = 0.1
    orientation_weight: float = -5.0
    action_rate_weight: float = -0.01
    max_tilt_radians: float = 1.0471975512
    command: VelocityCommandConfig = field(default_factory=VelocityCommandConfig)
    route: RouteCommandConfig = field(default_factory=RouteCommandConfig)
    gait: GaitConfig = field(default_factory=GaitConfig)
    terrain: TerrainConfig = field(default_factory=TerrainConfig)
    camera: DepthConfig = field(default_factory=DepthConfig)
    randomization: RandomizationConfig = field(default_factory=RandomizationConfig)
    training_reward: TrainingRewardConfig = field(default_factory=TrainingRewardConfig)
