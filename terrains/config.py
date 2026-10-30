"""集中管理连续混合路线、可变台阶高度和静态碎石参数。"""

from dataclasses import dataclass


@dataclass
class TerrainConfig:
    seed: int = 17
    levels: int = 3
    initial_level: int = 0
    layouts: int = 1
    obstacles: bool = True
    width: float = 3.0
    flat_length: float = 3.0
    step_count: int = 4
    step_depth: float = 0.35
    step_height_min: float = 0.03
    step_height_max: float = 0.09
    platform_length: float = 1.2
    transition_length: float = 1.2
    rubble_length: float = 3.0
    finish_length: float = 2.5
    rubble_rows: int = 5
    rubble_columns: int = 5
    rubble_height_min: float = 0.01
    rubble_height_max: float = 0.04
    spawn_x: float = 1.0

    @property
    def length(self):
        return (self.flat_length + 2 * self.step_count * self.step_depth
                + self.platform_length + self.transition_length
                + self.rubble_length + self.finish_length)

    def step_height(self, difficulty):
        return self.step_height_min + difficulty * (
            self.step_height_max - self.step_height_min
        )

    def regions(self):
        """区域区间均使用单条路线的局部 x，不能直接当作世界坐标。"""
        lengths = [
            ("flat", self.flat_length),
            ("stairs_up", self.step_count * self.step_depth),
            ("platform", self.platform_length),
            ("stairs_down", self.step_count * self.step_depth),
            ("transition", self.transition_length),
            ("rubble", self.rubble_length),
            ("finish", self.finish_length),
        ]
        result = {}
        start = 0.0
        for name, length in lengths:
            result[name] = (start, start + length)
            start += length
        return result
