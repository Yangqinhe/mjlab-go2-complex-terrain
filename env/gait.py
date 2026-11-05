"""记录真实接触步态；用无腾空、对角交替落足约束低速行走，不增加actor输入。"""

import torch


def flat_mask(task, local_x):
    cfg = task.config.terrain
    mask = torch.ones_like(local_x, dtype=torch.bool) if not cfg.obstacles else torch.zeros_like(local_x, dtype=torch.bool)
    if cfg.obstacles:
        for name, (start, end) in cfg.regions().items():
            if name in ('flat', 'platform', 'transition', 'finish'):
                margin = task.config.gait.flat_margin
                mask |= (local_x > start+margin) & (local_x < end-margin)
    # reset时机器人会短暂落地，不把初始化落地过程算作步态腾空。
    return mask & (task.env.episode_length_buf*task.dt > .5)


class GaitTracker:
    def __init__(self, config, num_envs, device, dt):
        self.config, self.dt = config, dt
        self.air_time = torch.zeros(num_envs, 4, device=device)
        self.previous_contact = torch.zeros(num_envs, 4, dtype=torch.bool, device=device)
        self.flight_time = torch.zeros(num_envs, device=device)
        self.max_flight = torch.zeros_like(self.flight_time)
        # 列依次为：样本、腾空、支撑足数、错误双足组合、双足样本、vz²、高度、高度²。
        self.sums = torch.zeros(num_envs, 8, device=device)
        self.touchdowns = torch.zeros_like(self.air_time)
        # 整条路线：样本、腾空、当前连续腾空秒数、最长连续腾空秒数。
        self.total = torch.zeros(num_envs, 4, device=device)

    def reset(self, ids):
        for value in (self.air_time, self.previous_contact, self.flight_time,
                      self.max_flight, self.sums, self.touchdowns, self.total):
            value[ids] = 0

    def step(self, contact, flat, vertical_velocity, height, active=None):
        cfg = self.config
        count = contact.sum(-1)
        flight = count == 0
        active = torch.ones_like(flat) if active is None else active
        self.total[:, 0] += active
        self.total[:, 1] += active & flight
        self.total[:, 2] = torch.where(active & flight, self.total[:, 2]+self.dt, 0.)
        self.total[:, 3] = torch.maximum(self.total[:, 3], self.total[:, 2])
        diagonal = ((contact[:, 0] & contact[:, 3]) | (contact[:, 1] & contact[:, 2])) & (count == 2)
        wrong = (count == 2) & ~diagonal
        touchdown = contact & ~self.previous_contact
        swing = self.air_time.clone()
        self.air_time = torch.where(contact, 0., self.air_time+self.dt)
        self.previous_contact.copy_(contact)
        self.flight_time = torch.where(flat & flight, self.flight_time+self.dt, 0.)
        self.max_flight = torch.maximum(self.max_flight, self.flight_time)
        values = torch.stack([torch.ones_like(height), flight.float(), count.float(),
                              wrong.float(), (count == 2).float(), vertical_velocity.square(),
                              height, height.square()], -1)
        self.sums += values*flat[:, None]
        self.touchdowns += (touchdown & (swing >= cfg.minimum_swing))*flat[:, None]
        if not cfg.enabled:
            return {}
        # 允许三/四足支撑；两足支撑时奖励对角组合，惩罚前后腿成组跳及同侧支撑。
        return dict(gait_flight=cfg.flight*flight.float()+cfg.flat_flight*(flat & flight),
                    gait_support=flat*(cfg.low_support*(count == 1)+cfg.wrong_pair*wrong
                                       +cfg.diagonal_support*diagonal),
                    gait_vertical=cfg.vertical*flat*vertical_velocity.square(),
                    gait_height=cfg.height*flat*(height-cfg.nominal_height).square(),
                    gait_touchdown=flat*(cfg.touchdown*(swing-cfg.minimum_swing).clamp(0, .2)*touchdown).sum(-1),
                    gait_short_swing=flat*(cfg.short_swing*(cfg.minimum_swing-swing).clamp(min=0)
                                          *touchdown*(swing > 0)).sum(-1))

    def snapshot(self, index):
        return dict(sums=self.sums[index].cpu().tolist(), max_flight=self.max_flight[index].item(),
                    touchdowns=self.touchdowns[index].cpu().tolist(),
                    all_samples=self.total[index, 0].item(), all_flight=self.total[index, 1].item(),
                    all_max_flight=self.total[index, 3].item())


def summarize_gait(records, dt=.02):
    sums = torch.tensor([r['gait']['sums'] for r in records]).sum(0)
    count = max(sums[0].item(), 1.)
    seconds = count*dt
    touchdowns = torch.tensor([r['gait']['touchdowns'] for r in records]).sum(0)
    total_samples = sum(r['gait']['all_samples'] for r in records)
    return dict(flat_samples=int(sums[0]), flight_fraction=sums[1].item()/count,
                all_samples=int(total_samples),
                all_flight_fraction=sum(r['gait']['all_flight'] for r in records)/max(total_samples, 1.),
                all_max_flight=max(r['gait']['all_max_flight'] for r in records),
                mean_contacts=sums[2].item()/count,
                wrong_pair_fraction=sums[3].item()/max(sums[4].item(), 1.),
                vertical_rms=(sums[5]/count).sqrt().item(),
                height_std=(sums[7]/count-(sums[6]/count).square()).clamp(min=0).sqrt().item(),
                max_flight=max(r['gait']['max_flight'] for r in records),
                touchdown_hz_by_foot=(touchdowns/seconds).tolist())


def gait_passed(report):
    g = report.get('gait', {})
    return (g.get('flat_samples', 0) >= 100 and g.get('all_samples', 0) >= 100
            and g['all_flight_fraction'] <= .01 and g['all_max_flight'] <= .040001
            and g['flight_fraction'] <= .01
            and g['max_flight'] <= .040001 and g['mean_contacts'] >= 1.9
            and g['wrong_pair_fraction'] <= .1 and g['vertical_rms'] <= .15
            and g['height_std'] <= .025
            and all(.35 <= hz <= 3. for hz in g['touchdown_hz_by_foot']))


def gait_error(report):
    """选模时优先减少步态违规，避免仅凭更低速度误差回退到跳跃策略。"""
    if not report.get('walking_gait'):
        return 0.
    g = report['gait']
    limits = dict(flight_fraction=.01, max_flight=.040001, wrong_pair_fraction=.1,
                  all_flight_fraction=.01, all_max_flight=.040001,
                  vertical_rms=.15, height_std=.025)
    error = sum(max(g.get(key, 1.)/limit-1, 0.) for key, limit in limits.items())
    error += max(1.9-g['mean_contacts'], 0.)/1.9
    error += sum(max(.35-hz, 0.)/.35+max(hz-3., 0.)/3. for hz in g['touchdown_hz_by_foot'])
    return error if g['flat_samples'] >= 100 else error+100.
