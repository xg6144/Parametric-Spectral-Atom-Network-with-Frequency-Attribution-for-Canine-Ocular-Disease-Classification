"""Matched-dense robustness check (2026-10-16), requested by an external re-review. No model is trained.

The image-quality degradations of experiments.robustness (Figure 9) are applied to
  shared preset, seed 42 (the models of Figure 9): the matched dense control results/training/canine_gfnet_matched_s42 is
      added; PSAN (AD 0.2) is evaluated again and must reproduce results/tables/robustness.csv;
  validation-selected, seeds 42-44: the c2 finals of PSAN (AD 0.2), the matched dense control and the free rank-16 filter.
Corruptions, severities, test images and the per-image noise draw (corruption seed 42 for every model) are those of
experiments.robustness, so every model receives exactly the same degraded images. Before any degradation, the clean
Macro F1 of every model must equal the test Macro F1 stored with its run. Outputs: results/round6_robustness/.

    $PY -m experiments.round6_robustness --device cuda:3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from evaluation_metrics import compute_metric_bundle
from . import round6 as r6
from .common import ROOT, load_manifest, load_model, sha256, write_json
from .robustness import SEVERITIES, CorruptionDataset
from .round4_analysis import selected_runs

OUT = ROOT / 'results/round6_robustness'
CORRUPTION_SEED = 42
SHARED = {'psan': 'results/training/canine_psan_gabor_ad0p2_M16_morlet_s42',
          'gfnet_matched': 'results/training/canine_gfnet_matched_s42'}
TUNED_ARMS = ['psan', 'gfnet_matched', 'lowrank16']
SEEDS = [42, 43, 44]
CORRUPTIONS = [('gaussian_noise', 'Gaussian noise'), ('gaussian_blur', 'Gaussian blur'), ('jpeg_quality', 'JPEG compression'),
               ('brightness_factor', 'Brightness'), ('contrast_factor', 'Contrast'), ('saturation_factor', 'Saturation')]


def load_any(directory):
    summary = json.loads((ROOT / directory / 'summary.json').read_text())
    checkpoint = ROOT / summary['checkpoint']
    if sha256(checkpoint) != summary['checkpoint_sha256']:
        raise ValueError('Checkpoint changed: ' + str(checkpoint))
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
    cfg = ckpt['config']
    if cfg.get('revision_protocol'):
        model, _, source = load_model(checkpoint)
    else:
        model = r6.make_model(cfg)
        source = next(k for k in ('eval_model', 'ema', 'model') if ckpt.get(k) is not None)
        model.load_state_dict(ckpt[source], strict=True)
    return model.eval(), summary, source


def evaluate(device):
    models, meta = {}, {}
    for arm, directory in SHARED.items():
        model, summary, source = load_any(directory)
        key = f'shared/{arm}/42'
        models[key] = model.to(device)
        meta[key] = dict(protocol='shared preset', arm=arm, seed=42, run=directory, state=source,
                         checkpoint_sha256=summary['checkpoint_sha256'], stored_test_macro_f1=summary['test']['macro_f1'])
    runs = selected_runs()
    for arm in TUNED_ARMS:
        for seed in SEEDS:
            directory = str(Path(runs[(arm, seed)]['checkpoint']).parent)
            model, summary, source = load_any(directory)
            key = f'tuned/{arm}/{seed}'
            models[key] = model.to(device)
            meta[key] = dict(protocol='validation-selected', arm=arm, seed=seed, run=directory, state=source,
                             checkpoint_sha256=summary['checkpoint_sha256'], stored_test_macro_f1=summary['test']['macro_f1'])
    frame = load_manifest()
    frame = frame[frame.split == 'test'].reset_index(drop=True)
    stored = pd.read_csv(ROOT / 'results/robustness/sample_manifest.csv')
    if not (stored.path.tolist() == frame.path.tolist() and stored.label.tolist() == frame.label.tolist()):
        raise ValueError('Test images differ from those of the original robustness test')
    classes = sorted(frame['class'].unique())
    conditions = [('clean', 0)] + [(name, level) for name in SEVERITIES for level in (1, 2, 3)]
    (OUT / 'probabilities').mkdir(parents=True, exist_ok=True)
    rows, clean = [], {}
    for name, level in conditions:
        probabilities, labels = {k: [] for k in models}, []
        loader = DataLoader(CorruptionDataset(frame, name, level, CORRUPTION_SEED), batch_size=128, num_workers=6,
                            shuffle=False)
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device); labels.append(y.numpy())
                for key, model in models.items():
                    probabilities[key].append(model(x).softmax(-1).float().cpu().numpy())
        y = np.concatenate(labels)
        if not np.array_equal(y, frame.label.to_numpy()):
            raise ValueError('Label order changed')
        for key, values in probabilities.items():
            p = np.concatenate(values)
            np.savez_compressed(OUT / 'probabilities' / f"{key.replace('/', '_')}_{name}_s{level}.npz",
                                probabilities=p.astype(np.float16), labels=y.astype(np.int8))
            metric = compute_metric_bundle(y, p, classes)['metrics']
            if name == 'clean':
                clean[key] = metric
                if abs(metric['macro_f1'] - meta[key]['stored_test_macro_f1']) > 1e-9:
                    raise ValueError(f"Clean Macro F1 of {key} ({metric['macro_f1']}) differs from its stored test result "
                                     f"({meta[key]['stored_test_macro_f1']})")
            rows.append(dict(model=key, protocol=meta[key]['protocol'], arm=meta[key]['arm'], seed=meta[key]['seed'],
                             corruption=name, severity=level, parameter=0 if name == 'clean' else SEVERITIES[name][level - 1],
                             images=len(y), accuracy=metric['accuracy'], macro_f1=metric['macro_f1'],
                             macro_auc_ovr=metric['macro_auc_ovr'],
                             macro_f1_absolute_drop=clean[key]['macro_f1'] - metric['macro_f1'],
                             macro_f1_relative_drop_percent=100 * (clean[key]['macro_f1'] - metric['macro_f1']) / clean[key]['macro_f1']))
        print('condition', name, level, 'done', flush=True)
        pd.DataFrame(rows).to_csv(OUT / 'robustness_matched.csv', index=False)
    return pd.DataFrame(rows), meta


def reproduction(table):
    """Shared-preset PSAN, evaluated again, against the stored Figure-9 values."""
    old = pd.read_csv(ROOT / 'results/tables/robustness.csv')
    old = old[old.model == 'psan'].set_index(['corruption', 'severity']).macro_f1
    new = table[table.model == 'shared/psan/42'].set_index(['corruption', 'severity']).macro_f1
    diff = (new - old.reindex(new.index)).abs()
    return dict(conditions=int(len(diff)), max_abs_difference=float(diff.max()), identical_to_1e9=int((diff <= 1e-9).sum()))


def summaries(table):
    tuned = table[table.protocol == 'validation-selected']
    drops = tuned[tuned.corruption != 'clean']
    by_arm = (drops.groupby(['corruption', 'severity', 'arm']).macro_f1_relative_drop_percent
              .agg(['mean', 'min', 'max']).reset_index())
    pivot = drops.pivot_table(index=['corruption', 'severity', 'seed'], columns='arm',
                              values='macro_f1_relative_drop_percent').reset_index()
    rows = []
    for (corruption, severity), part in pivot.groupby(['corruption', 'severity']):
        for other in ['gfnet_matched', 'lowrank16']:
            delta = (part['psan'] - part[other]).to_numpy()
            rows.append(dict(corruption=corruption, severity=severity, comparison=f'psan minus {other}',
                             drop_difference_mean=float(delta.mean()), drop_difference_min=float(delta.min()),
                             drop_difference_max=float(delta.max()), seeds_psan_smaller_drop=int((delta < 0).sum()),
                             seeds=len(delta)))
    return by_arm, pd.DataFrame(rows)


def figure(table):
    """Figure 9 with the matched dense control added (shared preset, seed 42)."""
    old = pd.read_csv(ROOT / 'results/tables/robustness.csv')
    dense = table[table.model == 'shared/gfnet_matched/42'].assign(model='gfnet_matched')
    frame = pd.concat([old, dense[old.columns.intersection(dense.columns)]], ignore_index=True)
    series = {'psan': ('PSAN', '#2a78d6', 'o', '-'), 'gfnet_matched': ('Matched dense', '#3d3d3a', 'D', '-.'),
              'gfnet': ('GFNet (repo)', '#eb6834', 's', '--'), 'afno': ('AFNO (repo)', '#1baf7a', '^', ':')}
    plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7,
                         'ytick.labelsize': 7, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.edgecolor': '#52514e',
                         'axes.labelcolor': '#0b0b0b', 'xtick.color': '#52514e', 'ytick.color': '#52514e'})
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 3.9), sharex=True)
    for ax, (key, title) in zip(axes.ravel(), CORRUPTIONS):
        for model, (label, color, marker, style) in series.items():
            clean = frame[(frame.model == model) & (frame.corruption == 'clean')].macro_f1.iloc[0]
            sub = frame[(frame.model == model) & (frame.corruption == key)].sort_values('severity')
            ax.plot([0] + sub.severity.tolist(), [clean] + sub.macro_f1.tolist(), color=color, marker=marker, ls=style,
                    ms=3.5, lw=1.4, label=label)
        ax.set_title(title, color='#0b0b0b'); ax.set_xticks([0, 1, 2, 3], ['clean', '1', '2', '3'])
        ax.grid(color='#e9e8e4', lw=.6)
        for side in ['top', 'right']:
            ax.spines[side].set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel('Test Macro F1')
    for ax in axes[1]:
        ax.set_xlabel('Severity')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, frameon=False, fontsize=7.5, bbox_to_anchor=(.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, .95))
    (OUT / 'figures').mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / 'figures' / 'robustness.pdf', bbox_inches='tight')
    fig.savefig(OUT / 'figures' / 'robustness.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--reuse', action='store_true', help='rebuild summaries and figure from the saved table')
    args = parser.parse_args()
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.reuse:
        table, meta = pd.read_csv(OUT / 'robustness_matched.csv'), json.loads((OUT / 'provenance.json').read_text())['models']
    else:
        table, meta = evaluate(torch.device(args.device))
    check = reproduction(table)
    if check['max_abs_difference'] > 1e-3:
        raise SystemExit('Shared-preset PSAN does not reproduce the stored robustness results: ' + json.dumps(check))
    by_arm, differences = summaries(table)
    by_arm.to_csv(OUT / 'robustness_tuned_by_arm.csv', index=False)
    differences.to_csv(OUT / 'robustness_tuned_differences.csv', index=False)
    figure(table)
    write_json(OUT / 'provenance.json', dict(
        models=meta, corruption_seed=CORRUPTION_SEED, severities=SEVERITIES, reproduction_of_figure9_psan=check,
        note='inference only; identical degraded images for every model; descriptive (no bootstrap intervals)',
        source_sha256={p: sha256(ROOT / p) for p in ['experiments/round6_robustness.py', 'experiments/robustness.py']}))
    print('Saved', OUT, json.dumps(check))


if __name__ == '__main__':
    main()
