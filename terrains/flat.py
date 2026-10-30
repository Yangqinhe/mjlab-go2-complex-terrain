"""创建静态地面和平台长方体，统一保存 mjlab 地形几何。"""

import mujoco
from mjlab.terrains.terrain_generator import TerrainGeometry


def add_box(body, name, x, y, length, width, top, color, bottom=-0.1):
    geom = body.add_geom(
        name=name,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=(x, y, (top + bottom) / 2),
        size=(length / 2, width / 2, (top - bottom) / 2),
        rgba=color,
        friction=(0.8, 0.01, 0.001),
        condim=3,
    )
    return TerrainGeometry(geom=geom, color=color)
