"""创建等高单段上下楼梯；不同路线的单级高度由难度决定。"""

from .flat import add_box


def add_stairs(body, config, difficulty, prefix):
    height = config.step_height(difficulty)
    regions = config.regions()
    geometries = []
    for direction, region in [("up", "stairs_up"), ("down", "stairs_down")]:
        start = regions[region][0]
        for index in range(config.step_count):
            # 上行最后一级与平台等高，下行最后一级回到地面。
            level = index + 1 if direction == "up" else config.step_count - index - 1
            if level == 0:
                continue  # 末级直接复用地面，避免叠放重合的碰撞表面。
            geometries.append(add_box(
                body, f"{prefix}_{direction}_{index}",
                start + (index + 0.5) * config.step_depth, config.width / 2,
                config.step_depth, config.width, level * height,
                (0.30, 0.52, 0.72, 1.0) if direction == "up"
                else (0.75, 0.48, 0.28, 1.0),
            ))
    start, end = regions["platform"]
    geometries.append(add_box(
        body, f"{prefix}_platform", (start + end) / 2, config.width / 2,
        end - start, config.width, config.step_count * height,
        (0.35, 0.65, 0.50, 1.0),
    ))
    return geometries
