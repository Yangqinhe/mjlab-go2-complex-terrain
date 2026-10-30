"""集中管理网络、PPO 和蒸馏训练参数。"""

from dataclasses import dataclass


# 台阶上限(m)、关节目标偏移尺度(rad)；前置档通过后才进入下一档。
STAIR_STAGES = ((.09, .5), (.12, .6), (.15, .7), (.18, .8), (.20, .9))


@dataclass
class TrainConfig:
    horizon: int = 32
    epochs: int = 5
    batch_size: int = 1024
    actor_lr: float = 1e-4
    critic_lr: float = 1e-3
    gamma: float = 0.99
    lmbda: float = 0.95
    epsilon: float = 0.2
    entropy_coef: float = 0.001
    target_kl: float = 0.015
    velocity_coef: float = 0.01
    kl_interval: int = 1
    history_length: int = 10
    history_dims: tuple[int, ...] = (450, 256, 128, 64)
    actor_dims: tuple[int, ...] = (176, 256, 128, 12)
    critic_dims: tuple[int, ...] = (244, 256, 128, 1)
    height_dims: tuple[int, ...] = (187, 128, 64)
    velocity_dims: tuple[int, ...] = (64, 32, 3)
    initial_std: float = 0.3
    save_interval: int = 100


@dataclass
class DistillConfig:
    num_envs: int = 256
    visual_steps: int = 20
    burn_in: int = 10
    sequence_batch: int = 16
    visual_lr: float = 1e-3
    common_lr: float = 1e-4
    latent_coef: float = 0.1
    velocity_coef: float = 0.1
