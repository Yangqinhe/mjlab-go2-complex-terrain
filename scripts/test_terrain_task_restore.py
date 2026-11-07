"""复核从宽平地恢复复杂路线时，通道宽度与地形宽度同步恢复。"""

from env.commands import free_training_config
from env.config import EnvConfig
from scripts.train_student_multitask import task_configs


def main():
    source = free_training_config(EnvConfig(), d435i_scenes=True)
    for strength in (0., 1.):
        route, free = task_configs(source, strength, 4, 4, 56000)
        low, high = route.terrain.corridor_width
        # 曾继承12～16m通道到3m路线，使邻列墙体穿过起点；检查实际尺寸关系。
        assert 0 < low <= high < route.terrain.width
        assert route.terrain.obstacles and route.terrain.layouts == 4
        assert free.terrain.corridor_width == (12., 16.) and free.terrain.width == 20.
        assert route.randomization.enabled == (strength > 0)
    assert source.terrain.width == 20. and source.terrain.corridor_width == (12., 16.)
    print('TERRAIN_TASK_RESTORE_PASSED')


if __name__ == '__main__':
    main()
