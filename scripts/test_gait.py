"""用对角交替、前后腿成组跳、腾空、站立和碎步案例核对步态约束。"""

import torch

from env.config import GaitConfig
from env.gait import GaitTracker, summarize_gait, gait_passed, gait_error


def check_pattern(pattern, period):
    tracker = GaitTracker(GaitConfig(enabled=True), 1, 'cpu', .02)
    reward = 0.
    for step in range(200):
        contact = torch.tensor([pattern[(step//period) % len(pattern)]], dtype=torch.bool)
        terms = tracker.step(contact, torch.tensor([True]), torch.zeros(1), torch.tensor([.27]))
        reward += sum(terms.values()).item()
    report = dict(walking_gait=True, gait=summarize_gait([dict(gait=tracker.snapshot(0))]))
    return gait_passed(report), reward, report


def main():
    diagonal = [[1, 0, 0, 1], [0, 1, 1, 0]]
    good, reward, report = check_pattern(diagonal, 15)
    bound, bound_reward, _ = check_pattern([[1, 1, 0, 0], [0, 0, 1, 1]], 15)
    jumping, _, _ = check_pattern([[1, 1, 1, 1], [0, 0, 0, 0]], 15)
    standing, _, _ = check_pattern([[1, 1, 1, 1]], 15)
    shuffling, _, _ = check_pattern(diagonal, 2)
    assert good and reward > bound_reward
    assert gait_error(report) == 0
    assert not any((bound, jumping, standing, shuffling))
    print(dict(status='passed', diagonal=report['gait'], rejects_bound=True,
               rejects_flight=True, rejects_standing=True, rejects_shuffling=True))


if __name__ == '__main__':
    main()
