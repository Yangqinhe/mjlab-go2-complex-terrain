"""米制光轴深度的固定缩放；缺失射线按远端处理，不增加age/valid特征。"""

import torch


def normalize_depth(depth, config):
    depth = torch.nan_to_num(depth, nan=config.far, posinf=config.far, neginf=config.far)
    return ((depth.clamp(config.near, config.far)-config.near)/
            (config.far-config.near)-.5).half()
