"""验证负载惯性、物理有效性、指令/时延与部分reset隔离，避免假随机化。"""

import json
from pathlib import Path

import torch
from mjlab.utils.lab_api.math import matrix_from_quat
from domain_randomization.dynamics import combine_payload
from env.training_task import Go2Task
from env.perception.runtime import VisualRuntime
from rl.run_io import env_from_dict, load_policy, save_json


def main():
    torch.set_num_threads(1)
    folder = Path('outputs/20261106_domain_randomization_setup')
    checkpoint = 'outputs/20261106_135343_676539_student_ppo_seed92037/checkpoints/final.pt'
    device = 'cuda:0'
    mass = torch.tensor([5., 5.], device=device)
    com = torch.zeros(2, 3, device=device)
    inertia = torch.diag(torch.tensor([.1, .2, .25], device=device)).expand(2, 3, 3)
    payload = torch.tensor([0., 1.], device=device)
    position = torch.tensor([[0., 0., .1], [.05, 0., .12]], device=device)
    total, center, diagonal, quat = combine_payload(mass, com, inertia, payload, position)
    assert torch.allclose(total, torch.tensor([5., 6.], device=device))
    assert torch.allclose(center[1], position[1]/6)
    rotation = matrix_from_quat(quat)
    restored = rotation@torch.diag_embed(diagonal)@rotation.transpose(-1, -2)
    assert torch.allclose(restored[0], inertia[0], atol=1e-6)
    assert (diagonal > 0).all()
    c = env_from_dict(json.loads((folder/'free_config.json').read_text()))
    actor, original, _ = load_policy(checkpoint, device)
    assert not original.randomization.enabled and original.command.mode == 'legacy'
    task = Go2Task(c, device)
    try:
        visual = VisualRuntime(task, actor.visual_encoder)
        p = task.proprio().clone()
        assert torch.equal(p, task.proprio()), '同周期重复调用不能重新采噪声'
        for i in range(80):
            with torch.no_grad():
                states, critic = task.inputs(visual.latent)
                _, terminated, truncated, _ = task.step(actor(states)[0])
                visual.update()
            ids = (terminated | truncated).nonzero(as_tuple=False).flatten()
            if len(ids):
                task.reset(ids); visual.reset(ids)
        body = task.env.sim.mj_model.body('robot/base').id
        observed = task.env.sim.model.body_mass[:, body].clone()
        assert (observed > 0).all() and observed.std() > .01
        assert task.env.sim.model.body_inertia[:, body].min() > 0
        contacts = [task.env.sim.mj_model.geom('robot/'+leg).id for leg in ('FL','FR','RL','RR')]
        assert (task.env.sim.model.geom_priority[contacts] == 1).all()
        assert task.env.sim.model.geom_friction[:, contacts[0], 0].std() > .01
        h = task.history[1:].clone(); v = visual.hidden[1:].clone()
        queue = task.randomization.queue[:, 1:].clone(); pairs = visual.pair[1:].clone()
        pending = visual.ready[1:].clone(); props = task.env.sim.model.body_mass[1:, body].clone()
        task.reset(torch.tensor([0], device=device));visual.reset(torch.tensor([0], device=device))
        assert torch.equal(h, task.history[1:]) and torch.equal(v, visual.hidden[1:])
        assert torch.equal(queue, task.randomization.queue[:, 1:])
        assert torch.equal(pairs, visual.pair[1:]) and torch.equal(pending, visual.ready[1:])
        assert torch.equal(props, task.env.sim.model.body_mass[1:, body])
        # 0/1周期的目标延迟，且上一执行动作与实际输入PD的归一化目标一致。
        runtime = task.randomization
        runtime.queue.zero_();runtime.cursor = 0;runtime.delay[:] = 1
        x = torch.full_like(task.previous_action, .2)
        assert runtime.action(x).abs().max() == 0
        assert torch.equal(runtime.action(x*2), x)
        task.schedule.anchor[:] = False
        for _ in range(5):task.schedule.sample(torch.arange(c.num_envs, device=device))
        assert task.schedule.target[:, 0].min() < 0 and task.schedule.target[:, 0].max() > 0
        assert task.schedule.target[:, 1].abs().max() == 0
        assert task.schedule.target[:, 2].abs().max() <= .5
        task.schedule.applied[:, 0] = .3;task.schedule.target.zero_();task.schedule.remaining[:] = 1000
        for _ in range(100):task.schedule.step()
        assert task.schedule.applied.abs().max() < 1e-5
        assert task.numerical_failures == 0
        save_json(folder/'verification.json', dict(status='passed',
            checks=['old_checkpoint_defaults','payload_parallel_axis','inertia_positive',
                    'per_world_mass_and_friction','foot_priority','one_noise_sample_per_step',
                    'partial_reset_isolation','action_delay','continuous_signed_commands','command_stop_ramp'],
            payload_mass_range=task.env.dr_samples[:,0].aminmax()[0].item(),
            numerical_failures=task.numerical_failures))
        print('DOMAIN_RANDOMIZATION_CHECKS_PASSED', flush=True)
    finally:
        task.close()


if __name__ == '__main__':
    main()
