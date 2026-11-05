"""从录像旁的50Hz接触记录导出落足时序、机身高度和速度，辅助检查步态。"""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace', type=Path)
    args = parser.parse_args()
    with args.trace.open() as stream:
        rows = list(csv.DictReader(stream))
    t = np.array([float(r['seconds']) for r in rows])
    contact = np.array([[int(r[leg]) for r in rows] for leg in ('FL', 'FR', 'RL', 'RR')])
    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True, constrained_layout=True)
    axes[0].imshow(contact, aspect='auto', interpolation='nearest', cmap='Greens',
                   vmin=0, vmax=1, extent=[t[0], t[-1], 3.5, -.5])
    axes[0].set(yticks=range(4), yticklabels=['FL', 'FR', 'RL', 'RR'],
                title='Foot contact: green = stance, white = swing')
    axes[1].plot(t, [float(r['height']) for r in rows])
    axes[1].set(ylabel='Base height above ground (m)')
    axes[2].plot(t, [float(r['vx']) for r in rows], label='Actual vx')
    axes[2].axhline(.3, color='gray', linestyle='--', label='Command 0.3 m/s')
    axes[2].set(xlabel='Time (s)', ylabel='Forward velocity (m/s)')
    axes[2].legend()
    for axis in axes[1:]:
        axis.grid(alpha=.2)
    output = args.trace.with_suffix('.png')
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(output)


if __name__ == '__main__':
    main()
