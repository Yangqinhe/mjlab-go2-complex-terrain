"""组装 Go2 与连续混合地形；返回 mjlab 原生批量环境供自写 PPO 后续接入。"""

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs.mdp import bad_orientation, time_out
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.viewer import ViewerConfig

from terrains.mixed_terrain import make_terrain_config
from . import observations, rewards
from .robot import make_robot_config


def make_env_config(config):
    command = config.command
    return ManagerBasedRlEnvCfg(
        seed=config.seed,
        decimation=config.decimation,
        episode_length_s=config.episode_seconds,
        # 后续自写 PPO 应先读取最终观测，再对 done 环境显式 reset，避免超时 bootstrap 串回合。
        auto_reset=False,
        scene=SceneCfg(
            num_envs=config.num_envs,
            terrain=make_terrain_config(config.terrain),
            entities={"robot": make_robot_config(config)},
        ),
        sim=SimulationCfg(
            nconmax=128, njmax=512,
            mujoco=MujocoCfg(timestep=config.physics_dt, cone="elliptic", impratio=10.0),
        ),
        commands={"velocity": UniformVelocityCommandCfg(
            entity_name="robot", heading_command=False, rel_heading_envs=0.0,
            resampling_time_range=(command.resample_seconds, command.resample_seconds),
            ranges=UniformVelocityCommandCfg.Ranges(
                lin_vel_x=command.vx, lin_vel_y=command.vy, ang_vel_z=command.wz,
            ),
        )},
        observations={"actor": ObservationGroupCfg(
            terms={"state": ObservationTermCfg(func=observations.policy_observation)},
            enable_corruption=False,
        )},
        actions={"joint_pos": JointPositionActionCfg(
            entity_name="robot", actuator_names=(".*",),
            scale=config.action_scale, use_default_offset=True,
            # 此 clip 作用于缩放后的关节目标，不能拿它当 [-1,1] 原始动作限幅。
            clip={".*_hip_joint": (-1.0472, 1.0472),
                  "F[LR]_thigh_joint": (-1.5708, 3.4907),
                  "R[LR]_thigh_joint": (-0.5236, 4.5379),
                  ".*_calf_joint": (-2.7227, -0.83776)},
        )},
        rewards={
            "tracking_xy": RewardTermCfg(
                func=rewards.tracking_velocity, weight=config.tracking_weight,
                params={"error_scale": config.tracking_error_scale},
            ),
            "orientation": RewardTermCfg(func=rewards.orientation_error,
                                         weight=config.orientation_weight),
            "action_rate": RewardTermCfg(func=rewards.action_rate,
                                         weight=config.action_rate_weight),
        },
        terminations={
            "tilt": TerminationTermCfg(func=bad_orientation,
                                       params={"limit_angle": config.max_tilt_radians}),
            "time_out": TerminationTermCfg(func=time_out, time_out=True),
        },
        viewer=ViewerConfig(entity_name="robot", body_name="base"),
    )


def create_env(config, device="cuda:0", render_mode=None):
    """配置一次传入；不在此加载 checkpoint、创建运行目录或执行训练。"""
    return ManagerBasedRlEnv(make_env_config(config), device=device, render_mode=render_mode)
