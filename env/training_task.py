"""训练任务适配：45维本体、精确地形、244维critic与显式终止/reset。"""

import math

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.utils.nan_guard import NanGuard

from env.go2env import make_env_config
from env.commands import route_command, VelocitySchedule
from env.gait import GaitTracker, flat_mask
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
        self.foot_radius = torch.tensor(self.env.sim.mj_model.geom_size[self.foot_ids, 0],
                                       device=device, dtype=torch.float32)
        names = self.env.scene["feet"].primary_names
        self.foot_order = [list(names).index(leg) for leg in ("FL", "FR", "RL", "RR")]
        self.calf_ids, _ = self.robot.find_bodies([f"{leg}_calf" for leg in ("FL", "FR", "RL", "RR")])
        x, y = torch.meshgrid(torch.linspace(-.3, 1.3, 17, device=device),
                              torch.linspace(-.5, .5, 11, device=device), indexing="ij")
        self.scan_xy = torch.stack([x.flatten(), y.flatten()], -1)
        # 特权教师可模拟相机采样周期；critic仍使用每个控制步的真实高度。
        self.height_cache = (torch.zeros(config.num_envs, len(self.scan_xy), device=device)
                             if config.height_scan_update_steps > 1 else None)
        self.height_cache_valid = torch.zeros(config.num_envs, dtype=torch.bool, device=device)
        self.height_cache_tick = -1
        self.history = torch.zeros(config.num_envs, 10, 45, device=device)
        self.previous_action = torch.zeros(config.num_envs, 12, device=device)
        self.previous_foot = torch.zeros(config.num_envs, 4, 3, device=device)
        self.return_sum = torch.zeros(config.num_envs, device=device)
        self.error_sum = torch.zeros(config.num_envs, device=device)
        self.vx_error_sum = torch.zeros_like(self.error_sum)
        self.visited = torch.zeros(config.num_envs, 7, dtype=torch.bool, device=device)
        self.dt = config.physics_dt*config.decimation
        self.numerical_failures = 0
        self.gait = GaitTracker(config.gait, config.num_envs, device, self.dt)
        self.schedule = VelocitySchedule(self) if config.command.mode != 'legacy' else None
        self.randomization = None
        if config.randomization.enabled:
            from domain_randomization.runtime import RandomizationRuntime
            self.randomization = RandomizationRuntime(self)
        self.yaw_error_sum = torch.zeros_like(self.error_sum)
        self.reset()

    def command(self):
        command = (self.schedule.applied if self.schedule is not None else
                   self.env.command_manager.get_command("velocity"))
        if self.config.route.enabled:
            r = self.robot.data
            q = r.root_link_quat_w
            yaw = torch.atan2(2*(q[:, 0]*q[:, 3]+q[:, 1]*q[:, 2]),
                              1-2*(q[:, 2].square()+q[:, 3].square()))
            lateral = r.root_link_pos_w[:, 1]-self.env.scene.env_origins[:, 1]
            command = route_command(command, lateral, yaw, self.config.route)
        return command

    def proprio(self):
        if self.randomization is not None:
            return self.randomization.observation
        return self.clean_proprio()

    def clean_proprio(self):
        r = self.robot.data
        command = self.command()
        return torch.cat([r.root_link_ang_vel_b*.25, r.projected_gravity_b,
                          command*torch.tensor([2., 2., .25], device=self.device),
                          r.joint_pos-r.default_joint_pos, r.joint_vel*.05,
                          self.previous_action], -1).clamp(-5, 5)
# self.robot.data：机器人在仿真环境中的状态数据。
# root_link_pos_w：机器人根链接（通常是机身）的世界坐标位置。
# root_link_quat_w：机器人根链接在世界坐标系下的四元数姿态。
    def inputs(self, visual_latent=None): #task.inputs()在训练循环中被调用，返回状态和critic输入
        r = self.robot.data
        pos = r.root_link_pos_w
        quat = r.root_link_quat_w
        yaw = torch.atan2(2*(quat[:, 0]*quat[:, 3]+quat[:, 1]*quat[:, 2]),
                          1-2*(quat[:, 2].square()+quat[:, 3].square()))
        c, s = yaw.cos()[:, None], yaw.sin()[:, None]
        xy = torch.stack([c*self.scan_xy[:, 0]-s*self.scan_xy[:, 1],
                          s*self.scan_xy[:, 0]+c*self.scan_xy[:, 1]], -1)+pos[:, None, :2]
        # 20cm×4级可累计0.8m；高阶允许±2，避免把高处台阶统一截成同一高度。
        clip = self.config.height_scan_clip #高度扫描裁剪值
        heights = ((self.ground(xy)-(pos[:, 2:3]-.27))/.5).clamp(-clip, clip)
        feet = self.env.sim.data.geom_xpos[:, self.foot_ids]
        foot_height = feet[:, :, 2]-self.ground(feet[:, :, :2]).clamp(min=0)
        contact = self.env.scene["feet"].data.force[:, self.foot_order, 2].abs()>1
        progress = ((pos[:, 0]-self.env.scene.env_origins[:, 0]) /
                    (self.config.terrain.length-self.config.terrain.spawn_x)).clamp(-1, 1)
        p = self.proprio()
        actor_heights = heights
        if visual_latent is None and self.height_cache is not None:
            tick = self.env.common_step_counter//self.config.height_scan_update_steps
            if tick != self.height_cache_tick:
                self.height_cache.copy_(heights)
                self.height_cache_tick = tick
            else:
                self.height_cache.copy_(torch.where(self.height_cache_valid[:, None], self.height_cache, heights))
            self.height_cache_valid.fill_(True)
            actor_heights = self.height_cache
        states = torch.cat([p, self.history.flatten(1),
                            actor_heights if visual_latent is None else visual_latent], -1)
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
        self.height_cache_valid[ids] = False
        self.previous_action[ids] = 0
        self.yaw_error_sum[ids] = 0
        if self.schedule is not None:
            self.schedule.reset(ids)
        if self.randomization is not None:
            self.randomization.reset(ids)
            self.randomization.update(self.clean_proprio(), ids)
        self.history[ids] = self.proprio()[ids, None].expand(-1, 10, -1)
        self.previous_foot[ids] = self.env.sim.data.geom_xpos[ids][:, self.foot_ids]
        self.return_sum[ids], self.error_sum[ids], self.visited[ids] = 0, 0, False
        self.vx_error_sum[ids] = 0
        self.gait.reset(ids)

    def step(self, action):
        executed = action.clamp(-1, 1)
        if self.randomization is not None:
            executed = self.randomization.action(executed)
        # command在整个transition内保持相同；终止状态入rollout后才为下一步换指令。
        command = self.command().clone()
        action_rate = (executed-self.previous_action).square().sum(-1)
        self.env.control_effort.zero_()
        self.env.step(executed)
        numerical = NanGuard.detect_nans(self.env.sim.data)
        if numerical.any():
            ids = numerical.nonzero(as_tuple=False).flatten()
            self.numerical_failures += len(ids)
            print(f'NUMERICAL_FAILURE env_ids={ids.tolist()} total={self.numerical_failures}', flush=True)
            # NaN比较不会触发姿态/高度终止。先恢复物理量，防止污染相机和末状态critic；
            # 下方仍明确记作失败终止，调用方照常reset并降课程，绝不bootstrap这个新状态。
            lengths = self.env.episode_length_buf[ids].clone()
            self.env.reset(env_ids=ids)
            self.env.episode_length_buf[ids] = lengths
        self.previous_action.copy_(executed)
        r = self.robot.data
        if self.schedule is None:
            command = self.command()
        feet = self.env.sim.data.geom_xpos[:, self.foot_ids]
        contact = self.env.scene["feet"].data.force[:, self.foot_order, 2].abs()>1
        raw_ground = self.ground(feet[:, :, :2])
        # 真正离开整张地图时不把-10缺失哨兵当支撑面；越界由body位置终止。
        ground = raw_ground.clamp(min=0)
        support = (ground*contact).sum(-1)/contact.sum(-1).clamp(min=1)
        support = torch.where(contact.any(-1), support, ground.median(-1).values)
        height = r.root_link_pos_w[:, 2]-support
        foot_velocity = self.robot.data.body_link_lin_vel_w[:, self.calf_ids]
        omega = self.robot.data.body_link_ang_vel_w[:, self.calf_ids]
        lever = feet-self.robot.data.body_link_pos_w[:, self.calf_ids]
        foot_velocity = foot_velocity+torch.cross(omega, lever, dim=-1)
        slip = (contact*foot_velocity[:, :, :2].square().sum(-1)).mean(-1)
        if self.weights.foot_landing:
            from env.posture import foot_velocities_world
            local_x = r.root_link_pos_w[:, 0]-self.env.scene.env_origins[:, 0]+self.config.terrain.spawn_x
            supported = self.env.scene['feet'].data.force[:, self.foot_order, 2].abs()>self.config.gait.contact_force
            point_velocity = foot_velocities_world(self, flat_contact_point=True)
            flat_slip = (supported*point_velocity[..., :2].square().sum(-1)).mean(-1)
            # 与平地审计一致，惩罚足底接触点估计速度；粗糙地形继续使用原足端中心近似。
            slip = torch.where(flat_mask(self, local_x), flat_slip, slip)
        error = (r.root_link_lin_vel_b[:, :2]-command[:, :2]).square().sum(-1)
        angular = r.root_link_ang_vel_b
        limits = r.soft_joint_pos_limits
        limit_error = ((limits[:, :, 0]-r.joint_pos).clamp(min=0)+
                       (r.joint_pos-limits[:, :, 1]).clamp(min=0)).square().sum(-1)
        collision = self.env.scene["nonfoot"].data.force.norm(dim=-1).amax(-1)>10
        w = self.weights
        # 移动足应越过前方台阶；静止支撑足速度近零，不强制腾空或指定步态相位。
        ahead = feet[:, :, :2].clone()
        ahead[:, :, 0] += w.lookahead_distance
        target = w.swing_clearance+(self.ground(ahead)-ground).clamp(0, .25)
        clearance = feet[:, :, 2]-ground-self.foot_radius
        moving = foot_velocity[:, :, :2].norm(dim=-1).sqrt()
        clearance_error = ((clearance-target).square()*moving*(raw_ground > -1)).sum(-1)
        terms = {
            "tracking_xy": w.tracking_xy*torch.exp(-error/self.config.tracking_error_scale),
            "tracking_yaw": w.tracking_yaw*torch.exp(-(angular[:, 2]-command[:, 2]).square()/w.tracking_yaw_scale),
            "progress": w.progress*r.root_link_lin_vel_w[:, 0].clamp(-.5, .5),
            "orientation": w.orientation*r.projected_gravity_b[:, :2].square().sum(-1),
            "angular_xy": w.angular_xy*angular[:, :2].square().sum(-1),
            "vertical": w.vertical*r.root_link_lin_vel_b[:, 2].square(),
            "height": w.height*(height-w.nominal_height).square(),
            "effort": w.effort*self.env.control_effort/self.config.decimation,
            "action_rate": w.action_rate*action_rate, "joint_limit": w.joint_limit*limit_error,
            "slip": w.slip*slip, "collision": w.collision*collision.float(),
            "foot_clearance": w.foot_clearance*clearance_error,
        }
        if w.front_hip_deviation or w.front_stance_width:
            from env.posture import front_posture_terms
            terms.update(front_posture_terms(self, command, flat_mask(self, local_x=(
                r.root_link_pos_w[:, 0]-self.env.scene.env_origins[:, 0]+self.config.terrain.spawn_x))))
        if w.all_hip_deviation or w.foot_landing:
            from env.posture import normal_foot_terms
            local_x = r.root_link_pos_w[:, 0]-self.env.scene.env_origins[:, 0]+self.config.terrain.spawn_x
            terms.update(normal_foot_terms(self, command, flat_mask(self, local_x)))
        free = self.config.command.mode == 'free'
        stopping = command.abs().amax(-1) < .03
        if self.schedule is not None:
            if free:
                terms['progress'] = torch.zeros_like(error)
            else:
                terms['progress'] *= ~stopping
            terms['standing'] = stopping*(-w.standing_velocity*r.root_link_lin_vel_b[:, :2].square().sum(-1)
                                          -w.standing_joint_velocity*r.joint_vel.square().mean(-1)
                                          -w.standing_angular*angular[:,2].square())
        pos = r.root_link_pos_w-self.env.scene.env_origins
        if self.config.route.enabled:
            terms['route_lateral'] = w.route_lateral*pos[:, 1].square()
            terms['route_yaw'] = w.route_yaw*(angular[:, 2]-command[:, 2]).square()
        local_x = pos[:, 0]+self.config.terrain.spawn_x
        gait_contact = self.env.scene['feet'].data.force[:, self.foot_order, 2].abs() > self.config.gait.contact_force
        support_mask = None
        if self.config.gait.terrain_support_downstairs_only:
            start, end = self.config.terrain.regions()['stairs_down']
            # 提前半个机身长度准备换足，尾足离开末级前继续保持落足重叠。
            support_mask = (local_x > start-.2) & (local_x < end+.2)
        terms.update(self.gait.step(gait_contact, flat_mask(self, local_x) & ~numerical,
                                   r.root_link_lin_vel_b[:, 2], height,
                                   active=(self.env.episode_length_buf*self.dt > .5) & ~numerical,
                                   support_mask=support_mask))
        if self.schedule is not None:
            for name in ('gait_touchdown', 'gait_short_swing', 'gait_support'):
                if name in terms:
                    terms[name] *= ~stopping
        regions = self.config.terrain.regions()
        for i, (start, end) in enumerate(regions.values()):
            self.visited[:, i] |= (local_x >= start) & (local_x <= end)
        fall = numerical | (r.projected_gravity_b[:, 2] > -math.cos(self.config.max_tilt_radians)) | (height<w.minimum_height)
        outside = (pos[:, 1].abs()>self.config.terrain.width/2-.2) | (local_x<0)
        success = (local_x > self.config.terrain.length-.5) & self.visited.all(-1) & ~fall
        if free:
            outside |= local_x > self.config.terrain.length-.2
            success = torch.zeros_like(fall)
        terminated = fall | outside | success
        truncated = self.env.episode_length_buf >= self.env.max_episode_length
        self.yaw_error_sum += (angular[:, 2]-command[:, 2]).square()
        if free:
            # 生存到预算末尾与跟踪准确分开；不能把所有未摔倒都写成成功。
            vx_rmse = ((self.vx_error_sum+(r.root_link_lin_vel_b[:, 0]-command[:, 0]).square())
                       /self.env.episode_length_buf.clamp(min=1)).sqrt()
            yaw_rmse = (self.yaw_error_sum/self.env.episode_length_buf.clamp(min=1)).sqrt()
            success = truncated & ~fall & ~outside & (vx_rmse <= .1) & (yaw_rmse <= .15)
        for value in terms.values():
            value[numerical] = 0
        reward = self.dt*sum(terms.values())+w.success*success.float()+w.failure*(fall | outside).float()
        self.return_sum += reward
        self.error_sum += error
        self.vx_error_sum += (r.root_link_lin_vel_b[:, 0]-command[:, 0]).square()
        info = dict(terms=terms, success=success, fall=fall, outside=outside, numerical_failure=numerical,
                    progress=pos[:, 0], height=height, velocity=r.root_link_lin_vel_b[:, 0],
                    error=error, episode_return=self.return_sum,
                    vx_rmse=(self.vx_error_sum/self.env.episode_length_buf.clamp(min=1)).sqrt(),
                    rmse=(self.error_sum/self.env.episode_length_buf.clamp(min=1)).sqrt())
        info['command'] = command
        info['yaw_rmse'] = (self.yaw_error_sum/self.env.episode_length_buf.clamp(min=1)).sqrt()
        if self.schedule is not None:
            self.schedule.step()
        if self.randomization is not None:
            self.randomization.update(self.clean_proprio())
        self.history = torch.cat([self.history[:, 1:], self.proprio()[:, None]], 1)
        self.previous_foot.copy_(feet)
        return reward, terminated, truncated, info

    def close(self):
        self.env.close()
