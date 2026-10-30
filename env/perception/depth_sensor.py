"""仅在显式capture时执行MJWarp批量针孔射线，不注册50Hz自动相机。"""

import math

import mujoco_warp as mjwarp
import torch
import warp as wp
from mjlab.utils.lab_api.math import matrix_from_quat

from .config import DepthConfig
from .preprocessing import normalize_depth


class DepthSensor:
    def __init__(self, task, config=None):
        self.task, self.config = task, config or DepthConfig()
        cfg = self.config
        device, n = task.device, task.config.num_envs
        fy = cfg.height/(2*math.tan(math.radians(cfg.fovy_degrees)/2))
        v, u = torch.meshgrid(torch.arange(cfg.height, device=device),
                              torch.arange(cfg.width, device=device), indexing='ij')
        directions = torch.stack([(u-(cfg.width-1)/2)/fy, -(v-(cfg.height-1)/2)/fy,
                                   -torch.ones_like(u)], -1).float().flatten(0, 1)
        self.directions = directions/directions.norm(dim=-1, keepdim=True)
        theta = math.radians(cfg.pitch_down_degrees)
        self.rotation = torch.tensor([[0, math.sin(theta), -math.cos(theta)],
                                      [-1, 0, 0], [0, math.cos(theta), math.sin(theta)]],
                                     device=device, dtype=torch.float32)
        self.mount = torch.tensor(cfg.position, device=device)
        rays = cfg.height*cfg.width
        self.pnt = torch.empty(n, rays, 3, device=device)
        self.vec = torch.empty_like(self.pnt)
        self.dist = torch.empty(n, rays, device=device)
        self.geom = torch.empty(n, rays, device=device, dtype=torch.int32)
        self.normal = torch.empty_like(self.pnt)
        self.wp_pnt = wp.from_torch(self.pnt, dtype=wp.vec3)
        self.wp_vec = wp.from_torch(self.vec, dtype=wp.vec3)
        self.wp_dist = wp.from_torch(self.dist)
        self.wp_geom = wp.from_torch(self.geom)
        self.wp_normal = wp.from_torch(self.normal, dtype=wp.vec3)
        self.exclude = wp.full(rays, -1, dtype=wp.int32, device=device)
        self.group_mask = wp.types.vector(length=6, dtype=float)(-1, 0, 0, 0, 0, 0)
        model = task.env.sim.mj_model
        with wp.ScopedDevice(device):
            self.context = mjwarp.create_render_context(
                mjm=model, nworld=n, cam_active=[False]*model.ncam,
                render_rgb=False, render_depth=False, use_textures=False,
                use_shadows=False, enabled_geom_groups=[0],
            )
        self.capture_count = 0
        self.raw_depth = None

    @torch.no_grad()
    def capture(self):
        r = self.task.robot.data
        body_rotation = matrix_from_quat(r.root_link_quat_w)
        rotation = body_rotation@self.rotation
        origin = r.root_link_pos_w+(body_rotation@self.mount[:, None]).squeeze(-1)
        self.pnt.copy_(origin[:, None].expand_as(self.pnt))
        self.vec.copy_(torch.einsum('bij,pj->bpi', rotation, self.directions))
        with wp.ScopedDevice(self.task.device):
            mjwarp.refit_bvh(self.task.env.sim.wp_model, self.task.env.sim.wp_data, self.context)
            mjwarp.rays(self.task.env.sim.wp_model, self.task.env.sim.wp_data,
                        self.wp_pnt, self.wp_vec, self.group_mask, True, self.exclude,
                        self.wp_dist, self.wp_geom, self.wp_normal, self.context)
        # r是射线量程，乘局部光轴分量后才是常用的Z-depth。
        depth = self.dist*(-self.directions[:, 2])
        depth = torch.where(self.dist >= 0, depth, self.config.far)
        self.raw_depth = depth.view(-1, self.config.height, self.config.width)
        self.capture_count += 1
        return normalize_depth(self.raw_depth, self.config)
