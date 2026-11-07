"""按可观察地形安排辅助参考，不混入奖励统计的初始化宽限条件。"""

import torch

from env.gait import flat_terrain_mask


def terrain_reference_weight(task, local_x, transition_distance=0.):
    weight=(~flat_terrain_mask(task,local_x)).float()
    if not transition_distance or not task.config.terrain.obstacles:
        return weight
    margin=task.config.gait.flat_margin
    if transition_distance<=margin:
        raise ValueError('参考过渡距离必须大于平地判定余量')
    # 障碍前逐渐建立参考，离开后渐退；相邻障碍间取较高权重，避免反复跳变。
    span=transition_distance-margin
    for name in ('stairs_up','stairs_down','rubble'):
        start,end=task.config.terrain.regions()[name]
        before=((local_x-start+transition_distance)/span).clamp(0,1)
        after=((end+transition_distance-local_x)/span).clamp(0,1)
        weight=torch.maximum(weight,torch.minimum(before,after))
    return weight
