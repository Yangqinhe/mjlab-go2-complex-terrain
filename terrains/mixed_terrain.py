"""把平地、上下楼梯和碎石组合在同一个 mjlab 子地形内。"""

from dataclasses import dataclass, field

import numpy as np
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.terrain_generator import (
    SubTerrainCfg, TerrainGeneratorCfg, TerrainOutput,
)

from .config import TerrainConfig
from .flat import add_box
from .rubble import add_rubble
from .stairs import add_stairs


@dataclass(kw_only=True)
class MixedTerrainCfg(SubTerrainCfg):
    # 继承 mjlab 的配置是接入子地形所需，不额外建立自定义生成器框架。
    course: TerrainConfig = field(default_factory=TerrainConfig)
    layout_id: int = 0

    def function(self, difficulty, spec, rng):
        config = self.course
        body = spec.body("terrain")
        prefix = f"course_{round(difficulty * 1000):04d}"
        if config.layouts > 1:
            prefix += f"_v{self.layout_id}"
        geometries = [add_box(
            body, f"{prefix}_ground", config.length / 2, config.width / 2,
            config.length, config.width, 0.0, (0.65, 0.67, 0.68, 1.0),
        )]
        if config.obstacles:
            geometries += add_stairs(body, config, difficulty, prefix)
            geometries += add_rubble(body, config, difficulty, rng, prefix)
        return TerrainOutput(
            origin=np.array([config.spawn_x, config.width / 2, 0.0]),
            geometries=geometries,
        )


def make_terrain_config(config):
    if config.levels < 2:
        raise ValueError("至少保留两档地形，才能验证楼梯高度变化")
    if not 0 <= config.initial_level < config.levels:
        raise ValueError("initial_level 必须位于已有地形等级范围")
    return TerrainEntityCfg(
        terrain_type="generator",
        max_init_terrain_level=config.initial_level,
        textures=(),
        materials=(),
        terrain_generator=TerrainGeneratorCfg(
            seed=config.seed,
            size=(config.length, config.width),
            num_rows=config.levels,
            curriculum=True,
            color_scheme="height",
            sub_terrains={f"mixed_course_{i}": MixedTerrainCfg(course=config, layout_id=i)
                          for i in range(config.layouts)},
        ),
    )
