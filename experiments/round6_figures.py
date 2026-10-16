"""Round-6 figure: learning curves with the 100-epoch runs of seeds 42-44 (replaces the seed-42 panels of round 4).

Left: validation Macro F1 of the validation-selected 30-epoch finals (mean and range over seeds 42-44). Middle and right:
validation Macro F1 and training loss of the 100-epoch runs of PSAN, the dense control and the free rank-16 control
(mean and range over seeds 42-44; seed 42 of PSAN and the dense control from round 4). No model is run.

    $PY -m experiments.round6_figures
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from .common import ROOT, sha256, write_json
from .round4_figures import COLORS, NAMES, STYLES, tidy

FIGURES = ROOT / 'results/round6/figures'
LONG_ARMS = ['psan', 'gfnet_matched', 'lowrank16']


def long_runs():
    runs = {}
    for base in [ROOT / 'results/round4/long', ROOT / 'results/round6/long']:
        for path in base.glob('*/summary.json'):
            s = json.loads(path.read_text())
            if s['arm'] in LONG_ARMS:
                runs[(s['arm'], s['seed'])] = path.parent
    return runs


def main():
    curves = pd.read_csv(ROOT / 'results/round4/tables/learning_curves.csv')
    short = curves[curves.budget == '30 epochs']
    runs = long_runs()
    if sorted(runs) != sorted((a, s) for a in LONG_ARMS for s in (42, 43, 44)):
        raise SystemExit('100-epoch runs incomplete: ' + str(sorted(runs)))
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.5), gridspec_kw=dict(width_ratios=[1, 1.45, 1.45]))
    for arm in [a for a in ['psan', 'psan_ad0', 'gfnet_matched', 'lowrank16'] if a in set(short.arm)]:
        sub = short[short.arm == arm].pivot_table(index='epoch', columns='seed', values='validation_macro_f1')
        axes[0].plot(sub.index, sub.mean(axis=1), STYLES[arm], color=COLORS[arm], linewidth=1.4, label=NAMES[arm])
        axes[0].fill_between(sub.index, sub.min(axis=1), sub.max(axis=1), color=COLORS[arm], alpha=.12, linewidth=0)
    axes[0].set(title='30 epochs (seeds 42–44)', xlabel='Epoch', ylabel='Validation Macro F1', ylim=(.3, .72))
    for arm in LONG_ARMS:
        frames = {s: pd.read_csv(runs[(arm, s)] / 'epochs.csv').set_index('epoch') for s in (42, 43, 44)}
        for ax, column in [(axes[1], 'validation_macro_f1'), (axes[2], 'train_loss')]:
            table = pd.DataFrame({s: f[column] for s, f in frames.items()})
            ax.plot(table.index, table.mean(axis=1), STYLES[arm], color=COLORS[arm], linewidth=1.4)
            ax.fill_between(table.index, table.min(axis=1), table.max(axis=1), color=COLORS[arm], alpha=.12, linewidth=0)
    axes[1].set(title='100 epochs (seeds 42–44)', xlabel='Epoch', ylabel='Validation Macro F1', ylim=(.3, .72))
    axes[2].set(title='100 epochs (seeds 42–44)', xlabel='Epoch', ylabel='Training loss')
    axes[2].set_yscale('log')
    for ax in axes:
        tidy(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(labels), frameon=False, bbox_to_anchor=(.5, 1.07))
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / 'learning_curves.pdf', bbox_inches='tight')
    fig.savefig(FIGURES / 'learning_curves.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    write_json(FIGURES / 'learning_curves_provenance.json', dict(
        runs={f'{a}/{s}': str(d.relative_to(ROOT)) for (a, s), d in sorted(runs.items())},
        short_source='results/round4/tables/learning_curves.csv',
        source_sha256={p: sha256(ROOT / p) for p in ['experiments/round6_figures.py', 'experiments/round4_figures.py']}))
    print('Saved', FIGURES / 'learning_curves.pdf')


if __name__ == '__main__':
    main()
