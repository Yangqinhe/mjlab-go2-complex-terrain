"""集中管理深度相机尺寸、外参、量程和更新周期。"""

from dataclasses import dataclass


@dataclass
class DepthConfig:
    width: int = 64
    height: int = 48
    fovy_degrees: float = 60.
    fovx_degrees: float | None = None  # None保留旧方形像素；实机缩放后允许fx/fy不同。
    pitch_down_degrees: float = 25.
    position: tuple[float, float, float] = (.27, 0., .10)
    near: float = .15
    far: float = 3.
    update_steps: int = 5
