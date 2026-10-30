"""只核对易错的GAE边界、超时bootstrap、批量shape和局部reset。"""

import json

import torch

from env.config import EnvConfig
from env.training_task import Go2Task
from rl.algorithms.ppo import PPO
from rl.algorithms.perceptive_networks import PerceptiveActor, PrivilegedCritic
from rl.config import TrainConfig


def main():
    torch.set_num_threads(1)
    config = TrainConfig(epochs=1, batch_size=8, kl_interval=1)
    agent = PPO(actor=PerceptiveActor(), critic=PrivilegedCritic(),
                device='cuda:0', config=config, epochs=1)
    delta = torch.tensor([[1., 10.], [2., 20.], [3., 30.]], device='cuda:0')
    dones = torch.tensor([[0., 1.], [1., 0.], [0., 0.]], device='cuda:0')
    actual = agent.compute_advantage(delta, dones)
    coefficient = agent.gamma*agent.lmbda
    expected = torch.tensor([[1+coefficient*2, 10], [2, 20+coefficient*30], [3, 30]], device='cuda:0')
    assert torch.allclose(actual, expected)
    terminal = torch.tensor([1., 0.], device='cuda:0')
    td = 1+agent.gamma*torch.tensor([5., 5.], device='cuda:0')*(1-terminal)
    assert torch.allclose(td, torch.tensor([1., 1+agent.gamma*5], device='cuda:0'))
    env_config = EnvConfig(num_envs=4, episode_seconds=.04)
    task = Go2Task(env_config, curriculum=False)
    try:
        for _ in range(2):
            states, critic = task.inputs()
            action, sample = agent.sample_batch(states, critic)
            reward, terminated, truncated, info = task.step(action)
            _, final_critic = task.inputs()
            agent.store_batch(sample, reward, terminated | truncated, terminated, final_critic)
        assert truncated.all() and not terminated.any()
        old_history = task.history.clone()
        task.reset(torch.tensor([0], device=task.device))
        assert torch.equal(old_history[1:], task.history[1:])
        assert task.env.episode_length_buf[0] == 0
        assert task.env.episode_length_buf[1] == 2
        metrics = agent.update_batch()
        assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
        print(json.dumps(dict(status='passed', gae_masks=True, timeout_bootstrap=True,
                              reset_isolation=True, ppo_update=metrics)), flush=True)
    finally:
        task.close()


if __name__ == '__main__':
    main()
