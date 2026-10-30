"""从最终碰撞模型提取轴对齐box高度；不把简化地形表传给学生actor。"""

import numpy as np
import torch


class HeightQuery:
    def __init__(self, env, config):
        model = env.sim.mj_model
        table = []
        for level in range(config.levels):
            prefix = f"course_{round(level/(config.levels-1)*1000):04d}"
            for layout in range(config.layouts):
                tag = prefix + (f"_v{layout}" if config.layouts > 1 else "") + "_"
                boxes = [i for i in range(model.ngeom) if model.geom(i).name.startswith(tag)]
                positions, sizes = model.geom_pos[boxes], model.geom_size[boxes]
                assert np.allclose(model.geom_quat[boxes], [1, 0, 0, 0]), "Only axis-aligned boxes"
                table.append(np.c_[positions[:, :2]-sizes[:, :2],
                                   positions[:, :2]+sizes[:, :2], positions[:, 2]+sizes[:, 2]])
        self.boxes = torch.tensor(np.stack(table), device=env.device, dtype=torch.float32)
        self.terrain = env.scene.terrain
        self.layouts = config.layouts

    def __call__(self, xy):
        indices = self.terrain.terrain_levels*self.layouts+self.terrain.terrain_types
        boxes = self.boxes[indices]
        inside = ((xy[:, :, None] >= boxes[:, None, :, :2]) &
                  (xy[:, :, None] <= boxes[:, None, :, 2:4])).all(-1)
        return boxes[:, None, :, 4].expand_as(inside).masked_fill(~inside, -10.).amax(-1)
