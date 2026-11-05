"""用对角交替、前后腿成组跳、腾空、站立和碎步案例核对步态约束。"""

import torch

from env.config import GaitConfig
from env.gait import GaitTracker, summarize_gait, gait_passed, gait_error
from scripts.train_pipeline import passed, route_passed, selection_score


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
    # 平地正常、楼梯/碎石区腾空，也必须被整条路线的检查拒绝。
    tracker = GaitTracker(GaitConfig(enabled=True), 1, 'cpu', .02)
    for step in range(200):
        tracker.step(torch.tensor([diagonal[(step//15) % 2]], dtype=torch.bool),
                     torch.tensor([True]), torch.zeros(1), torch.tensor([.27]))
    for _ in range(5):
        tracker.step(torch.zeros(1, 4, dtype=torch.bool), torch.tensor([False]),
                     torch.zeros(1), torch.tensor([.27]))
    assert not gait_passed(dict(gait=summarize_gait([dict(gait=tracker.snapshot(0))])))
    # 非平地也计时，第三个20ms无支撑周期才触发；落地和reset都清除连续计时。
    tracker = GaitTracker(GaitConfig(enabled=True, flight=-10., long_flight=-20.), 1, 'cpu', .02)
    airborne = torch.zeros(1, 4, dtype=torch.bool)
    flat, velocity, height = torch.tensor([False]), torch.zeros(1), torch.tensor([.27])
    terms = [tracker.step(airborne, flat, velocity, height) for _ in range(3)]
    costs = [r['gait_long_flight'].item() for r in terms]
    assert [r['gait_flight'].item() for r in terms] == [-10.]*3
    assert costs == [0., 0., -20.]
    assert tracker.step(~airborne, flat, velocity, height)['gait_long_flight'].item() == 0.
    tracker.reset(torch.tensor([0]))
    assert tracker.step(airborne, flat, velocity, height)['gait_long_flight'].item() == 0.
    good_report = dict(report, partial_evaluation=False, unique_layouts=20, success_rate=1., vx_rmse=.03)
    stopped = dict(good_report, success_rate=0., vx_rmse=.3)
    repair = dict(good_report, gait=dict(report['gait'], all_max_flight=.1))
    assert passed(good_report) and not passed(repair)
    assert route_passed(repair) and not route_passed(stopped)
    unfinished = dict(good_report, success_rate=.7, vx_rmse=.14)
    assert route_passed(unfinished, min_success=.5, max_rmse=.2)
    assert not passed(unfinished) and not route_passed(unfinished)
    assert selection_score([good_report, good_report]) > selection_score([good_report, repair])
    assert selection_score([good_report, repair]) > selection_score([good_report, stopped])
    print(dict(status='passed', diagonal=report['gait'], rejects_bound=True,
               rejects_flight=True, rejects_standing=True, rejects_shuffling=True))


if __name__ == '__main__':
    main()
