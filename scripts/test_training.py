"""只核对易错的GAE边界、超时bootstrap、批量shape和局部reset。"""

import json
from dataclasses import asdict

import mujoco
import numpy as np
import torch

from env.config import EnvConfig, RouteCommandConfig
from env.commands import route_command
from env.training_task import Go2Task
from rl.algorithms.ppo import PPO
from rl.algorithms.perceptive_networks import PerceptiveActor, PrivilegedCritic
from rl.config import TrainConfig
from rl.run_io import restore_curriculum


def main():
    torch.set_num_threads(1)
    command = torch.tensor([[.3, 0., 0.]]*3)
    guided = route_command(command, torch.tensor([.5, -.5, 0.]), torch.zeros(3), RouteCommandConfig())
    assert guided[0, 2] < 0 < guided[1, 2] and guided[2, 2] == 0
    assert guided[:, 2].abs().max() <= .5 and torch.equal(guided[:, :2], command[:, :2])
    config = TrainConfig(epochs=1, batch_size=8, kl_interval=1,
                         adaptive_lr=True, kl_stop_multiplier=2.)
    agent = PPO(actor=PerceptiveActor(), critic=PrivilegedCritic(),
                device='cuda:0', config=config, epochs=1, actor_lr=3e-5)
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
    env_config.terrain.layouts = 4
    env_config.terrain.initial_level = 2
    env_config.route.enabled = True
    env_config.gait.enabled = True
    task = Go2Task(env_config, curriculum=False)
    try:
        # 跨过分配地块边缘后，邻块真实地面不能被当成-10m的空洞。
        xy = task.env.scene.env_origins[:, None, :2].clone()
        xy[:, :, 1] += 1.6
        height = task.ground(xy).flatten().cpu().numpy()
        model = task.env.sim.mj_model
        cpu = mujoco.MjData(model)
        mujoco.mj_forward(model, cpu)
        for index, point in enumerate(xy[:, 0].cpu().numpy()):
            ray = mujoco.mj_ray(model, cpu, np.array([*point, 2.]),
                np.array([0., 0., -1.]), np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8),
                1, -1, np.zeros(1, dtype=np.int32))
            assert abs(height[index]-(2-ray if ray >= 0 else -10)) < 1e-5
        for _ in range(2):
            states, critic = task.inputs()
            action, sample = agent.sample_batch(states, critic)
            reward, terminated, truncated, info = task.step(action)
            _, final_critic = task.inputs()
            agent.store_batch(sample, reward, terminated | truncated, terminated, final_critic)
        assert truncated.all() and not terminated.any()
        assert 'route_lateral' in info['terms'] and 'route_yaw' in info['terms']
        assert 'gait_support' in info['terms'] and 'gait_flight' in info['terms']
        assert torch.allclose(task.command()[:, 0], torch.full((4,), .3, device=task.device))
        old_history = task.history.clone()
        task.reset(torch.tensor([0], device=task.device))
        assert torch.equal(old_history[1:], task.history[1:])
        assert task.env.episode_length_buf[0] == 0
        assert task.env.episode_length_buf[1] == 2
        metrics = agent.update_batch()
        assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
        assert config.min_actor_lr <= metrics['actor_lr'] <= config.max_actor_lr
        # 数值异常必须算失败终止，不能让NaN进入下一步actor或影响其他世界。
        task.reset()
        original_step = task.env.step
        unaffected = None
        def inject_invalid(action):
            nonlocal unaffected
            result = original_step(action)
            unaffected = task.env.sim.data.qpos[1:].clone()
            task.env.sim.data.qvel[0, 0] = float('nan')
            return result
        task.env.step = inject_invalid
        try:
            states, critic = task.inputs()
            action, sample = agent.sample_batch(states, critic)
            reward, terminated, truncated, info = task.step(action)
        finally:
            task.env.step = original_step
        assert info['numerical_failure'].tolist() == [True, False, False, False]
        assert terminated[0] and info['fall'][0] and not info['success'][0]
        assert reward[0] == task.weights.failure
        assert torch.equal(task.env.sim.data.qpos[1:], unaffected)
        states, final_critic = task.inputs()
        assert torch.isfinite(states).all() and torch.isfinite(final_critic).all()
        agent.store_batch(sample, reward, terminated | truncated, terminated, final_critic)
        task.reset(torch.tensor([0], device=task.device), torch.tensor([False], device=task.device))
        assert all(torch.isfinite(torch.tensor(v)) for v in agent.update_batch().values())
        old_levels = task.env.scene.terrain.terrain_levels.clone()
        saved = dict(env_config=asdict(env_config), curriculum_levels=old_levels)
        task.env.scene.terrain.terrain_levels.zero_()
        restore_curriculum(task, saved)
        assert torch.equal(task.env.scene.terrain.terrain_levels, old_levels)
        print(json.dumps(dict(status='passed', gae_masks=True, timeout_bootstrap=True,
                              reset_isolation=True, neighbor_ground=True,
                              route_following=True,
                              numerical_failure_termination=True,
                              curriculum_restore=True, ppo_update=metrics)), flush=True)
    finally:
        task.close()


if __name__ == '__main__':
    main()
