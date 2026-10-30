"""从实验CSV导出PPT可用训练曲线；训练回合成功率与独立评估分开标注。"""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('metrics', type=Path)
    parser.add_argument('--window', type=int, default=20)
    args = parser.parse_args()
    with args.metrics.open() as stream:
        rows = list(csv.DictReader(stream))
    values = {key:np.array([float(row[key]) for row in rows]) for key in rows[0]}
    width = min(args.window, len(rows))
    smooth = lambda data: np.convolve(data, np.ones(width)/width, mode='valid')
    x = values['iteration'][width-1:]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for ax, key, title in zip(axes.flat,
            ['vx', 'tracking_rmse', 'reward', 'successes', 'critic_loss', 'mean_level'],
            ['Forward velocity (m/s)', 'Planar tracking RMSE (m/s)', 'Reward / policy step',
             'Training episode success rate (not held-out)', 'Critic MSE', 'Mean terrain level']):
        curve = smooth(values[key])
        if key == 'successes':
            curve = curve/np.maximum(smooth(values['episodes']), 1e-8)
        ax.plot(x, curve, linewidth=1.5)
        if key == 'vx':
            ax.axhline(.3, color='gray', linestyle='--', label='Command 0.3')
            ax.legend()
        ax.set(title=title, xlabel='PPO iteration')
        ax.grid(alpha=.2)
    fig.suptitle(args.metrics.parent.name)
    output = args.metrics.parent/'training_curves.png'
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(output)


if __name__ == '__main__':
    main()
