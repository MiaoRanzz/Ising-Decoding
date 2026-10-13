#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render raw budget-pilot BCE curves, cross-checked against train logs."""

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, choices=(20, 50), default=20)
    epochs = parser.parse_args().epochs
    tail_start = epochs - epochs // 5 + 1
    root = Path(__file__).resolve().parents[4]
    source = root / 'outputs/qewc_v4/budget_pilot/selection.json'
    data = json.loads(source.read_text())
    titles = {'T1': 'Measurement noise', 'T2': 'CNOT noise',
              'T3': 'Idle noise', 'T4': 'Z-biased noise'}
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.titleweight': 'bold'})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.4))
    for ax, task in zip(axes.flat, titles):
        row = next(r for r in data['tasks'][task]['records'] if r['legacy_epochs'] == epochs)
        train, val = row['train_bce_trajectory'], row['validation_bce_trajectory']
        assert len(train) == len(val) == epochs
        log_rows = re.findall(r'LOSS train ([0-9.eE+-]+) valid ([0-9.eE+-]+)',
                              Path(row['log']).read_text())
        assert [(float(t), float(v)) for t, v in log_rows] == list(zip(train, val))
        ax.axvspan(tail_start - 0.5, epochs + 0.5, color='#64748b', alpha=0.08)
        if epochs == 50 and task in ('T3', 'T4'):
            ax.axvline(49.5 if task == 'T3' else 46.5, color='#64748b',
                       linestyle=':', linewidth=1, label='Resume boundary')
        for values, label, color, marker in [(train, 'Train BCE', '#2563eb', 'o'),
                                              (val, 'Validation BCE', '#ea580c', 's')]:
            ax.plot(range(1, epochs + 1), values, label=label, color=color,
                    marker=marker, markersize=3.5, linewidth=1.7)
            ax.annotate(f'{values[-1]:.5f}', (epochs, values[-1]), xytext=(7, 0),
                        textcoords='offset points', color=color, va='center', fontsize=10)
        ax.set(title=f'{task} | {titles[task]}', xlabel='Epoch', ylabel='BCE loss',
               xlim=(0.5, epochs * 1.15))
        ax.set_xticks([1, 5, 10, 15, 20] if epochs == 20 else [1, 10, 20, 30, 40, 50])
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.5f'))
        ax.grid(alpha=0.2)
        ax.legend(loc='upper right', frameon=False, fontsize=9)
    fig.suptitle(f'QEWC v4 | HTNet A101 | E{epochs} training curves', fontsize=18, y=0.98)
    fig.text(0.5, 0.925, 'Independent T0-initialized task runs | seed 20260907 | raw epoch losses, no smoothing',
             ha='center', color='#475569', fontsize=11)
    fig.text(0.5, 0.025, f'Each panel uses its own non-zero y-axis range. Shading: epochs {tail_start}–{epochs}.\n'
             f'Source: E{epochs} train logs (BCE logged to 5 decimals); not per-epoch LER.',
             ha='center', color='#475569', fontsize=10)
    fig.tight_layout(rect=(0, 0.075, 1, 0.91), h_pad=2, w_pad=2)
    destination = root / 'experiment/qewc-v4/figures'
    destination.mkdir(parents=True, exist_ok=True)
    for extension in ('png', 'pdf'):
        path = destination / f'e{epochs}_training_curves.{extension}'
        fig.savefig(path, dpi=180, bbox_inches='tight')
        print(path)
    plt.close(fig)


if __name__ == '__main__':
    main()
