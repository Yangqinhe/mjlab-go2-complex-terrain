"""训练任务适配：45维本体、精确地形、244维critic与显式终止/reset。"""

import math

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from env.go2env import make_env_config
from terrains.height_query import HeightQuery


def accumulate_effort(env):
    value = env.scene["robot"].data.actuator_force.square().sum(-1)
    env.control_effort.add_(value)
    return value


class Go2Task:
    def __init__(self, config, device="cuda:0", curriculum=True):
        self.config, self.device, self.curriculum = config, device, curriculum
        self.weights = config.training_reward
        cfg = make_env_config(config)
        # 训练信号在本适配器中统一计算，不改变旧加载环境的默认接口。
        cfg.rewards, cfg.terminations = {}, {}
        cfg.scene.sensors = (
            ContactSensorCfg(name="feet", primary=ContactMatch(
                mode="geom", pattern=("FL", "FR", "RL", "RR"), entity="robot"),
                secondary=ContactMatch(mode="body", pattern="terrain"),
                fields=("force",), reduce="netforce"),
            ContactSensorCfg(name="nonfoot", primary=ContactMatch(
                mode="geom", pattern="collision_.*", entity="robot"),
                secondary=ContactMatch(mode="body", pattern="terrain"),
                fields=("force",), reduce="netforce"),
        )
        cfg.metrics = {"effort": MetricsTermCfg(func=accumulate_effort, per_substep=True)}
        cfg.sim.contact_sensor_maxmatch = 512
        self.env = ManagerBasedRlEnv(cfg, device=device)
        self.env.control_effort = torch.zeros(config.num_envs, device=device)
        self.robot = self.env.scene["robot"]
        self.ground = HeightQuery(self.env, config.terrain)
        self.foot_ids = [self.env.sim.mj_model.geom(f"robot/{leg}").id
                         for leg in ("FL", "FR", "RL", "RR")]
        names = self.env.scene["feet"].primary_names
        self.foot_order = [list(names).index(leg) for leg in ("FL", "FR", "RL", "RR")]
        self.calf_ids, _ = self.robot.find_bodies([f"{leg}_calf" for leg in ("FL", "FR", "RL", "RR")])
        x, y = torch.meshgrid(torch.linspace(-.3, 1.3, 17, device=device),
                              torch.linspace(-.5, .5, 11, device=device), indexing="ij")
        self.scan_xy = torch.stack([x.flatten(), y.flatten()], -1)
        self.history = torch.zeros(config.num_envs, 10, 45, device=device)
        self.previous_action = torch.zeros(config.num_envs, 12, device=device)
        self.previous_foot = torch.zeros(config.num_envs, 4, 3, device=device)
        self.return_sum = torch.zeros(config.num_envs, device=device)
        self.error_sum = torch.zeros(config.num_envs, device=device)
        self.vx_error_sum = torch.zeros_like(self.error_sum)
        self.visited = torch.zeros(config.num_envs, 7, dtype=torch.bool, device=device)
        self.dt = config.physics_dt*config.decimation
        self.reset()

    def proprio(self):
        r = self.robot.data
        command = self.env.command_manager.get_command("velocity")
        return torch.cat([r.root_link_ang_vel_b*.25, r.projected_gravity_b,
                          command*torch.tensor([2., 2., .25], device=self.device),
                          r.joint_pos-r.default_joint_pos, r.joint_vel*.05,
                          self.previous_action], -1).clamp(-5, 5)

    def inputs(self, visual_latent=None):
        r = self.robot.data
        pos = r.root_link_pos_w
        quat = r.root_link_quat_w
        yaw = torch.atan2(2*(quat[:, 0]*quat[:, 3]+quat[:, 1]*quat[:, 2]),
                          1-2*(quat[:, 2].square()+quat[:, 3].square()))
        c, s = yaw.cos()[:, None], yaw.sin()[:, None]
        xy = torch.stack([c*self.scan_xy[:, 0]-s*self.scan_xy[:, 1],
                          s*self.scan_xy[:, 0]+c*self.scan_xy[:, 1]], -1)+pos[:, None, :2]
        heights = ((self.ground(xy)-(pos[:, 2:3]-.27))/.5).clamp(-1, 1)
        feet = self.env.sim.data.geom_xpos[:, self.foot_ids]
        foot_height = feet[:, :, 2]-self.ground(feet[:, :, :2])
        contact = self.env.scene["feet"].data.force[:, self.foot_order, 2].abs()>1
        progress = ((pos[:, 0]-self.env.scene.env_origins[:, 0]) /
                    (self.config.terrain.length-self.config.terrain.spawn_x)).clamp(-1, 1)
        p = self.proprio()
        states = torch.cat([p, self.history.flatten(1),
                            heights if visual_latent is None else visual_latent], -1)
        critic = torch.cat([p, r.root_link_lin_vel_b, heights, contact.float(),
                            foot_height, progress[:, None]], -1)
        return states, critic

    def reset(self, ids=None, move_up=None):
        ids = torch.arange(self.config.num_envs, device=self.device) if ids is None else ids
        if not len(ids):
            return
        terrain = self.env.scene.terrain
        if move_up is not None and self.curriculum:
            terrain.update_env_origins(ids, move_up, ~move_up)
        self.env.reset(env_ids=ids)
        self.previous_action[ids] = 0
        self.history[ids] = self.proprio()[ids, None].expand(-1, 10, -1)
        self.previous_foot[ids] = self.env.sim.data.geom_xpos[ids][:, self.foot_ids]
        self.return_sum[ids], self.error_sum[ids], self.visited[ids] = 0, 0, False
        self.vx_error_sum[ids] = 0

    def step(self, action):
        executed = action.clamp(-1, 1)
        action_rate = (executed-self.previous_action).square().sum(-1)
        self.env.control_effort.zero_()
        self.env.step(executed)
        self.previous_action.copy_(executed)
        self.history = torch.cat([self.history[:, 1:], self.proprio()[:, None]], 1)
        r = self.robot.data
        feet = self.env.sim.data.geom_xpos[:, self.foot_ids]
        contact = self.env.scene["feet"].data.force[:, self.foot_order, 2].abs()>1
        ground = self.ground(feet[:, :, :2])
        support = (ground*contact).sum(-1)/contact.sum(-1).clamp(min=1)
        support = torch.where(contact.any(-1), support, ground.median(-1).values)
        height = r.root_link_pos_w[:, 2]-support
        foot_velocity = self.robot.data.body_link_lin_vel_w[:, self.calf_ids]
        omega = self.robot.data.body_link_ang_vel_w[:, self.calf_ids]
        lever = feet-self.robot.data.body_link_pos_w[:, self.calf_ids]
        foot_velocity = foot_velocity+torch.cross(omega, lever, dim=-1)
        slip = (contact*foot_velocity[:, :, :2].square().sum(-1)).mean(-1)
        command = self.env.command_manager.get_command("velocity")
        error = (r.root_link_lin_vel_b[:, :2]-command[:, :2]).square().sum(-1)
        angular = r.root_link_ang_vel_b
        limits = r.soft_joint_pos_limits
        limit_error = ((limits[:, :, 0]-r.joint_pos).clamp(min=0)+
                       (r.joint_pos-limits[:, :, 1]).clamp(min=0)).square().sum(-1)
        collision = self.env.scene["nonfoot"].data.force.norm(dim=-1).amax(-1)>10
        w = self.weights
        terms = {
            "tracking_xy": w.tracking_xy*torch.exp(-error/self.config.tracking_error_scale),
            "tracking_yaw": w.tracking_yaw*torch.exp(-(angular[:, 2]-command[:, 2]).square()/.25),
            "progress": w.progress*r.root_link_lin_vel_w[:, 0].clamp(-.5, .5),
            "orientation": w.orientation*r.projected_gravity_b[:, :2].square().sum(-1),
            "angular_xy": w.angular_xy*angular[:, :2].square().sum(-1),
            "vertical": w.vertical*r.root_link_lin_vel_b[:, 2].square(),
            "height": w.height*(height-w.nominal_height).square(),
            "effort": w.effort*self.env.control_effort/self.config.decimation,
            "action_rate": w.action_rate*action_rate, "joint_limit": w.joint_limit*limit_error,
            "slip": w.slip*slip, "collision": w.collision*collision.float(),
        }
        pos = r.root_link_pos_w-self.env.scene.env_origins
        local_x = pos[:, 0]+self.config.terrain.spawn_x
        regions = self.config.terrain.regions()
        for i, (start, end) in enumerate(regions.values()):
            self.visited[:, i] |= (local_x >= start) & (local_x <= end)
        fall = (r.projected_gravity_b[:, 2] > -math.cos(self.config.max_tilt_radians)) | (height<w.minimum_height)
        outside = (pos[:, 1].abs()>self.config.terrain.width/2-.2) | (local_x<0)
        success = (local_x > self.config.terrain.length-.5) & self.visited.all(-1) & ~fall
        terminated = fall | outside | success
        truncated = self.env.episode_length_buf >= self.env.max_episode_length
        reward = self.dt*sum(terms.values())+w.success*success.float()+w.failure*(fall | outside).float()
        self.return_sum += reward
        self.error_sum += error
        self.vx_error_sum += (r.root_link_lin_vel_b[:, 0]-command[:, 0]).square()
        info = dict(terms=terms, success=success, fall=fall, outside=outside,
                    progress=pos[:, 0], height=height, velocity=r.root_link_lin_vel_b[:, 0],
                    error=error, episode_return=self.return_sum,
                    vx_rmse=(self.vx_error_sum/self.env.episode_length_buf.clamp(min=1)).sqrt(),
                    rmse=(self.error_sum/self.env.episode_length_buf.clamp(min=1)).sqrt())
        self.previous_foot.copy_(feet)
        return reward, terminated, truncated, info

    def close(self):
        self.env.close()
