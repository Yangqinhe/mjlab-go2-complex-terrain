"""生成深度相机可见且可碰撞的侧墙/侧边物体；中央通行带保持畅通。"""

from .flat import add_box


def add_surroundings(body, config, rng, prefix, layout_id):
    # 四种结构按布局固定，尺寸/位置由几何seed采样；不在机器人面前凭空生成封路墙。
    mode = layout_id % 4  # 0开放，1左墙，2右墙，3双侧墙。
    gap = rng.uniform(*config.corridor_width)
    wall_top = config.step_count*config.step_height_max+rng.uniform(1.3, 2.0)
    start, end = rng.uniform(0., .8), config.length-rng.uniform(0., .5)
    geometries = []
    for side, enabled in [(-1, mode in (2, 3)), (1, mode in (1, 3))]:
        top, bottom = (wall_top, -.1) if enabled else (-1., -2.)
        item = add_box(body, f'{prefix}_scene_wall_{side}', (start+end)/2,
                       config.width/2+side*(gap/2+.06), end-start, .12, top,
                       (.64, .64, .60, 1.), bottom=bottom)
        item.geom.group = 4
        geometries.append(item)
    for index, side in enumerate((-1, 1)):
        size = rng.uniform(.15, .25)
        # 内侧边界至少离中线0.55m，避免把随机装饰变成阻断速度任务的障碍。
        lateral = max(.55+size/2, gap/2-size/2-.04)
        top = rng.uniform(.25, 1.0)
        item = add_box(body, f'{prefix}_scene_object_{index}',
                       rng.uniform(2., config.length-1.), config.width/2+side*lateral,
                       rng.uniform(.25, .60), size, top, (.45, .47, .50, 1.))
        item.geom.group = 4
        geometries.append(item)
    return geometries
