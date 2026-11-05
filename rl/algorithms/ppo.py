"""单环境 PPO-Clip；rollout 可跨回合，超时与跌倒使用不同的 bootstrap mask。"""
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence

from .ActorCritic import ActorNet, CriticNet


class PPO:
    def __init__(self, state_dim=58, hidden_dim=128, action_dim=12,
                 actor_lr=3e-4, critic_lr=1e-3, gamma=0.99, lmbda=0.95,
                 epochs=5, device='cpu', epsilon=0.2, actor=None, critic=None, config=None):
        self.device = device
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.epsilon = epsilon
        # 常改参数集中在这里；入口只暴露训练规模、指令和文件路径。
        self.batch_size = 256
        self.target_kl = 0.015
        self.entropy_coef = 0.001
        self.actor = (actor or ActorNet(state_dim, hidden_dim, action_dim)).to(device)
        self.critic = (critic or CriticNet(state_dim, hidden_dim)).to(device)
        self.config = config
        if config is not None:
            self.batch_size, self.target_kl = config.batch_size, config.target_kl
            self.entropy_coef = config.entropy_coef
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.clear_buffer()

    def clear_buffer(self):
        self.buffer = {key: [] for key in
                       ('states', 'actions', 'rewards', 'dones', 'terminated',
                        'values', 'next_values', 'log_probs', 'means', 'stds')}

    @torch.no_grad()
    def store_transition(self, state, action, reward, next_state, done, terminated):
        """必须在本轮 update 前调用；存储采样策略的概率和价值，更新时不重算 old。"""
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        next_tensor = torch.as_tensor(next_state, dtype=torch.float32, device=self.device)
        action_tensor = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        mu, std = self.actor(state_tensor)
        values = {
            'states': np.array(state, copy=True),
            'actions': np.array(action, copy=True),
            'rewards': reward, 'dones': done, 'terminated': terminated,
            'values': self.critic(state_tensor).item(),
            'next_values': self.critic(next_tensor).item(),
            'log_probs': Normal(mu, std).log_prob(action_tensor).sum().item(),
            'means': mu.cpu().numpy().copy(), 'stds': std.cpu().numpy().copy(),
        }
        for key, value in values.items():
            self.buffer[key].append(value)

    @torch.no_grad()
    def take_action(self, state, deterministic=False):
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        mu, std = self.actor(state)
        # 缓存和 log_prob 使用原始 Gaussian 样本；只有 env 将其限幅。
        # tanh 只用于均值，并没有对随机样本做 tanh 变换。
        action = mu if deterministic else Normal(mu, std).sample()
        return action.cpu().numpy()

    def compute_advantage(self, td_delta, dones):
        advantage = torch.zeros_like(td_delta)
        gae = 0.0
        for t in reversed(range(len(td_delta))):
            gae = td_delta[t] + self.gamma * self.lmbda * (1 - dones[t]) * gae
            advantage[t] = gae
        return advantage

    def update(self):
        if not self.buffer['states']:
            return None
        data = {key: torch.as_tensor(np.asarray(value), dtype=torch.float32,
                                     device=self.device)
                for key, value in self.buffer.items()}
        # 跌倒不 bootstrap；时间截断使用 reset 前最后观测的 V。
        # 两种回合结束都切断 GAE，rollout 边界则自然使用 next_value。
        td_delta = (data['rewards'] + self.gamma * data['next_values']
                    * (1 - data['terminated']) - data['values'])
        advantage = self.compute_advantage(td_delta, data['dones'])
        returns = advantage + data['values']
        advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)
        old_distribution = Normal(data['means'], data['stds'])
        actor_losses, critic_losses = [], []
        stop_actor = False
        count = len(returns)

        for _ in range(self.epochs):
            for indices in torch.randperm(count, device=self.device).split(self.batch_size):
                states = data['states'][indices]
                if not stop_actor:
                    # 对完整 rollout 测 KL，避免小 minibatch 的偶然偏差。
                    with torch.no_grad():
                        distribution = Normal(*self.actor(data['states']))
                        kl = kl_divergence(old_distribution, distribution).sum(-1).mean()
                        stop_actor = kl.item() > self.target_kl
                if not stop_actor:
                    distribution = Normal(*self.actor(states))
                    log_prob = distribution.log_prob(data['actions'][indices]).sum(-1)
                    ratio = (log_prob - data['log_probs'][indices]).exp()
                    surrogate = ratio * advantage[indices]
                    clipped = ratio.clamp(1 - self.epsilon, 1 + self.epsilon) * advantage[indices]
                    actor_loss = -torch.minimum(surrogate, clipped).mean()
                    actor_loss -= self.entropy_coef * distribution.entropy().sum(-1).mean()
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                    self.actor_optimizer.step()
                    actor_losses.append(actor_loss.item())

                # KL 只停止 actor，critic 仍继续拟合固定的 returns。
                critic_loss = F.mse_loss(self.critic(states).squeeze(-1), returns[indices])
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
                self.critic_optimizer.step()
                critic_losses.append(critic_loss.item())

        with torch.no_grad():
            distribution = Normal(*self.actor(data['states']))
            kl = kl_divergence(old_distribution, distribution).sum(-1).mean()
            ratio = (distribution.log_prob(data['actions']).sum(-1) - data['log_probs']).exp()
            metrics = {
                'actor_loss': float(np.mean(actor_losses)) if actor_losses else 0.0,
                'critic_loss': float(np.mean(critic_losses)), 'kl': kl.item(),
                'clip_fraction': ((ratio - 1).abs() > self.epsilon).float().mean().item(),
                'std': distribution.stddev.mean().item(),
                'action_clip_fraction': (data['actions'].abs() > 1).float().mean().item(),
                'actor_steps': len(actor_losses),
            }
        self.clear_buffer()
        return metrics

    @torch.no_grad()
    def sample_batch(self, states, critic_states, deterministic=False):
        """批量GPU采样；旧take_action仍返回NumPy，避免静默改变已有接口。"""
        mu, std = self.actor(states)
        action = mu if deterministic else Normal(mu, std).sample()
        return action, {
            'states': states.detach().clone(), 'critic_states': critic_states.detach().clone(),
            'actions': action, 'values': self.critic(critic_states).squeeze(-1),
            'log_probs': Normal(mu, std).log_prob(action).sum(-1),
            'means': mu, 'stds': std,
        }

    @torch.no_grad()
    def store_batch(self, sample, reward, done, terminated, next_critic):
        if not hasattr(self, 'batch_buffer'):
            self.batch_buffer = []
        sample.update(rewards=reward.clone(), dones=done.float(), terminated=terminated.float(),
                      next_values=self.critic(next_critic).squeeze(-1))
        self.batch_buffer.append(sample)

    @torch.no_grad()
    def batch_kl(self, data):
        total = torch.zeros((), device=self.device)
        for start in range(0, len(data['states']), 8192):
            end = start+8192
            distribution = Normal(*self.actor(data['states'][start:end]))
            old = Normal(data['means'][start:end], data['stds'][start:end])
            total += kl_divergence(old, distribution).sum()
        return total/len(data['states'])

    def update_batch(self):
        """保留原PPO-Clip/GAE/KL早停；先沿T计算GAE，再展开T*N。"""
        data = {key: torch.stack([row[key] for row in self.batch_buffer])
                for key in self.batch_buffer[0]}
        td_delta = (data['rewards']+self.gamma*data['next_values']*(1-data['terminated'])
                    -data['values'])
        advantage = self.compute_advantage(td_delta, data['dones'])
        returns = (advantage+data['values']).flatten()
        advantage = advantage.flatten()
        advantage = (advantage-advantage.mean())/(advantage.std(unbiased=False)+1e-8)
        data = {key: value.flatten(0, 1) for key, value in data.items()}
        stop_actor, steps = False, 0
        totals = torch.zeros(4, device=self.device)
        count = 0
        for _ in range(self.epochs):
            for ids in torch.randperm(len(returns), device=self.device).split(self.batch_size):
                if not stop_actor and steps % self.config.kl_interval == 0:
                    stop_actor = self.batch_kl(data).item() > self.target_kl*self.config.kl_stop_multiplier
                if not stop_actor:
                    distribution = Normal(*self.actor(data['states'][ids]))
                    ratio = (distribution.log_prob(data['actions'][ids]).sum(-1)
                             -data['log_probs'][ids]).exp()
                    surrogate = ratio*advantage[ids]
                    clipped = ratio.clamp(1-self.epsilon, 1+self.epsilon)*advantage[ids]
                    policy_loss = -torch.minimum(surrogate, clipped).mean()
                    policy_loss -= self.entropy_coef*distribution.entropy().sum(-1).mean()
                    velocity_loss = F.mse_loss(2*self.actor.velocity(data['states'][ids]),
                                               2*data['critic_states'][ids, 45:48])
                    actor_loss = policy_loss+self.config.velocity_coef*velocity_loss
                    if not torch.isfinite(actor_loss):
                        raise RuntimeError('Non-finite actor loss')
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), .5)
                    self.actor_optimizer.step()
                    totals[0] += policy_loss.detach()
                    totals[1] += velocity_loss.detach()
                    steps += 1
                critic_loss = F.mse_loss(self.critic(data['critic_states'][ids]).squeeze(-1),
                                         returns[ids])
                if not torch.isfinite(critic_loss):
                    raise RuntimeError('Non-finite critic loss')
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.)
                self.critic_optimizer.step()
                totals[2] += critic_loss.detach()
                count += 1
        with torch.no_grad():
            kl = self.batch_kl(data).item()
            # 按完整rollout的KL调整下一轮步长，保留早停；不靠取消约束增加更新次数。
            if self.config.adaptive_lr:
                for group in self.actor_optimizer.param_groups:
                    if kl > 1.5*self.target_kl:
                        group['lr'] = max(group['lr']/1.5, self.config.min_actor_lr)
                    elif kl < .5*self.target_kl:
                        group['lr'] = min(group['lr']*1.5, self.config.max_actor_lr)
            metrics = dict(actor_loss=totals[0].item()/max(steps, 1),
                           velocity_loss=totals[1].item()/max(steps, 1),
                           critic_loss=totals[2].item()/max(count, 1), kl=kl,
                           std=self.actor.log_std.clamp(getattr(self.actor, 'log_std_min', -3.), 0).exp().mean().item(),
                           action_clip_fraction=(data['actions'].abs()>1).float().mean().item(),
                           actor_steps=steps, actor_lr=self.actor_optimizer.param_groups[0]['lr'])
        self.batch_buffer.clear()
        return metrics
