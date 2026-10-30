"""创建有固定 seed 的静态碎石块体；此阶段不模拟可滚动石块。"""

from .flat import add_box


def add_rubble(body, config, difficulty, rng, prefix):
    start, end = config.regions()["rubble"]
    dx = (end - start) / config.rubble_rows
    dy = config.width / config.rubble_columns
    geometries = []
    maximum = config.rubble_height_min + difficulty * (
        config.rubble_height_max - config.rubble_height_min
    )
    for row in range(config.rubble_rows):
        for column in range(config.rubble_columns):
            x = start + (row + 0.5) * dx + rng.uniform(-0.08, 0.08) * dx
            y = (column + 0.5) * dy + rng.uniform(-0.08, 0.08) * dy
            height = rng.uniform(config.rubble_height_min, maximum)
            geometries.append(add_box(
                body, f"{prefix}_rubble_{row}_{column}", x, y,
                dx * rng.uniform(0.60, 0.85), dy * rng.uniform(0.60, 0.85),
                height, (0.48, 0.46, 0.43, 1.0),
            ))
    return geometries
