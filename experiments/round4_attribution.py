"""Seed stability of class-atom attributions (round 4, post hoc, requested by external review).

For the validation-selected PSAN (atom dropout 0.2, setting c2) of seeds 42-44, the first-order
atom attributions Attr_c(m) of the predicted-class logit are computed at blocks 0, 5 and 11 on a
stratified test sample of up to 200 images per class (pandas sampling with random_state=42), with the
same `attributions` routine as the faithfulness analysis. Correctly classified images give the
class-atom matrix of each seed (mean signed attribution per class, normalized to unit length, as in
the main figure). Atom indices are run-specific, so atoms are matched to seed 42 by the correlation of
their learned envelope magnitudes on the frequency grid (Hungarian assignment). A permutation-free
summary sums the attributions of the atoms centered within half a frequency bin of DC and of all
other atoms. No training is involved; inference only.

    $PY -m experiments.round4_attribution --device cuda:0
"""
from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.ticker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader

from .common import CLASSES, ROOT, ManifestDataset, build_model, load_manifest, seed_all, sha256, write_json
from .faithfulness import attributions
from .round3_figures import DIVERGING, LABELS, MUTED, atom_parameters, envelope

OUT = ROOT / 'results/round4/attribution_stability'
CHECKPOINTS = {42: 'results/tuning/final/canine_psan_c2_s42_final/best.pth',
               43: 'results/round3/tuning/final/canine_psan_c2_s43_final/best.pth',
               44: 'results/round3/tuning/final/canine_psan_c2_s44_final/best.pth'}
BLOCKS = [0, 5, 11]
PER_CLASS = 200
DC_RADIUS = np.pi / 14          # half of the 2*pi/14 bin spacing of the 14-token axis


def load_psan(path, device):
    checkpoint = torch.load(ROOT / path, map_location='cpu', weights_only=False)
    cfg = checkpoint['config']
    if cfg.get('model', 'psan') != 'psan' or cfg.get('atoms', 16) != 16:
        raise ValueError('Expected a PSAN M16 checkpoint: ' + path)
    model = build_model('psan', classes=11, size=224, atom='gabor', dropout=cfg.get('atom_dropout', .2), atoms=16, init='morlet')
    state = checkpoint['eval_model'] if checkpoint.get('eval_model') is not None else checkpoint['ema']
    model.load_state_dict({k: v for k, v in state.items() if not k.endswith(('.filter.U', '.filter.V'))}, strict=True)
    return model.to(device).eval()


def sample_frame():
    frame = load_manifest()
    test = frame[frame.split == 'test']
    chosen = [g.sample(n=min(len(g), PER_CLASS), random_state=42) for _, g in test.groupby('label')]
    return pd.concat(chosen).sort_values('path').reset_index(drop=True)


def class_matrix(values, labels, targets):
    rows, counts = [], []
    for k in range(len(CLASSES)):
        chosen = values[(labels == k) & (targets == k)]
        counts.append(int(len(chosen)))
        mean = chosen.mean(0) if len(chosen) else np.zeros(values.shape[1])
        rows.append(mean / max(np.linalg.norm(mean), 1e-12))
    return np.asarray(rows), counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    torch.set_num_threads(4); seed_all(42)
    device = torch.device(args.device)
    OUT.mkdir(parents=True, exist_ok=True)
    frame = sample_frame()
    frame.to_csv(OUT / 'sample_manifest.csv', index=False)
    loader = DataLoader(ManifestDataset(frame), batch_size=32, num_workers=4, shuffle=False)
    images = torch.cat([x for x, _ in loader])
    labels = frame.label.to_numpy(int)
    results, geometry = {}, {}
    for seed, path in CHECKPOINTS.items():
        model = load_psan(path, device)
        values = {b: [] for b in BLOCKS}; targets = []
        for start in range(0, len(images), 32):
            x = images[start:start + 32].to(device)
            with torch.no_grad():
                target = model(x).argmax(-1)
            targets.append(target.cpu().numpy())
            for block in BLOCKS:
                attr, *_ = attributions(model, x, block, target)
                values[block].append(attr.cpu().numpy())
        targets = np.concatenate(targets)
        results[seed] = dict(targets=targets, values={b: np.concatenate(v) for b, v in values.items()},
                             checkpoint_sha256=sha256(ROOT / path))
        geometry[seed] = {}
        for block in BLOCKS:
            p = atom_parameters(model.cpu(), block)
            env = envelope(p, p['grid_u'], p['grid_v']).reshape(16, -1)
            geometry[seed][block] = dict(radius=p['radius'], envelope=env)
        print('seed', seed, 'accuracy on sample', float((targets == labels).mean()), flush=True)

    rows, matches = [], []
    for seed in CHECKPOINTS:
        for block in BLOCKS:
            radius = geometry[seed][block]['radius']
            dc = radius < DC_RADIUS
            matrix, counts = class_matrix(results[seed]['values'][block], labels, results[seed]['targets'])
            for k, name in enumerate(CLASSES):
                chosen = results[seed]['values'][block][(labels == k) & (results[seed]['targets'] == k)]
                rows.append(dict(seed=seed, block=block, cls=name, correct=counts[k], dc_atoms=int(dc.sum()),
                                 dc_attribution=float(chosen[:, dc].sum(1).mean()) if len(chosen) else None,
                                 other_attribution=float(chosen[:, ~dc].sum(1).mean()) if len(chosen) else None,
                                 row=json.dumps([float(x) for x in matrix[k]])))
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / 'class_dc_summary.csv', index=False)
    for seed in (43, 44):
        for block in BLOCKS:
            a, b = geometry[42][block]['envelope'], geometry[seed][block]['envelope']
            similarity = np.corrcoef(a, b)[:16, 16:]
            similarity = np.nan_to_num(similarity, nan=0.)
            row_index, col_index = linear_sum_assignment(-similarity)
            reference, _ = class_matrix(results[42]['values'][block], labels, results[42]['targets'])
            other, _ = class_matrix(results[seed]['values'][block], labels, results[seed]['targets'])
            matched = other[:, col_index]
            per_class = [float(np.corrcoef(reference[k], matched[k])[0, 1]) for k in range(len(CLASSES))]
            dc_ref = summary[(summary.seed == 42) & (summary.block == block)].set_index('cls').dc_attribution
            dc_other = summary[(summary.seed == seed) & (summary.block == block)].set_index('cls').dc_attribution
            matches.append(dict(seed=seed, block=block, mean_envelope_correlation=float(similarity[row_index, col_index].mean()),
                                matrix_correlation=float(np.corrcoef(reference.ravel(), matched.ravel())[0, 1]),
                                median_class_row_correlation=float(np.median(per_class)),
                                dc_sign_agreement=int((np.sign(dc_ref) == np.sign(dc_other)).sum()),
                                dc_attribution_correlation=float(np.corrcoef(dc_ref, dc_other)[0, 1]),
                                assignment=json.dumps([int(c) for c in col_index])))
    pd.DataFrame(matches).to_csv(OUT / 'seed_matching.csv', index=False)

    # Figure: per-class DC and non-DC attribution in block 11 for each seed.
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.9), sharey=True)
    block11 = summary[summary.block == 11]
    y = np.arange(len(CLASSES))
    for ax, column, title in zip(axes, ['dc_attribution', 'other_attribution'],
                                 ['Atoms at the DC bin', 'All other atoms']):
        for offset, seed, color in [(-.25, 42, '#1f4e9c'), (0., 43, '#2a9d8f'), (.25, 44, '#c2410c')]:
            values = block11[block11.seed == seed].set_index('cls').loc[CLASSES, column].to_numpy(float)
            ax.barh(y + offset, values, height=.24, color=color, label=f'Seed {seed}')
        ax.axvline(0, color=MUTED, linewidth=.8)
        ax.set_title(title); ax.set_xlabel('Mean summed attribution (true-class logit)')
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4)); ax.tick_params(axis='x', labelsize=7)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
    axes[0].set_yticks(y, [f'{LABELS[c]}' for c in CLASSES]); axes[0].invert_yaxis()
    axes[1].legend(frameon=False, fontsize=7, loc='lower right')
    fig.tight_layout()
    for suffix in ['pdf', 'png']:
        fig.savefig(OUT / f'dc_attribution_seeds.{suffix}', bbox_inches='tight', dpi=300)
    plt.close(fig)

    # Figures: class-atom matrices of seed 42 on the larger sample (replace the 220-image versions).
    for block in BLOCKS:
        matrix, counts = class_matrix(results[42]['values'][block], labels, results[42]['targets'])
        order = np.argsort(geometry[42][block]['radius'])
        limit = np.ceil(np.abs(matrix).max() * 10) / 10
        fig = plt.figure(figsize=(7.16, 2.9))
        ax = fig.add_axes([.24, .12, .62, .70]); bar_ax = fig.add_axes([.875, .12, .014, .70])
        image = ax.imshow(matrix[:, order], cmap=DIVERGING, vmin=-limit, vmax=limit, aspect='auto', interpolation='nearest')
        ax.set_xticks(range(16), [str(m) for m in order])
        ax.set_yticks(range(len(CLASSES)), [f'{LABELS[c]} ({n})' for c, n in zip(CLASSES, counts)])
        ax.set_xlabel(f'Atom index $m$ (block {block}), ordered by learned center radius')
        top = ax.secondary_xaxis('top')
        top.set_xticks(range(16), [f'{geometry[42][block]["radius"][m]:.2f}' for m in order], fontsize=6)
        top.set_xlabel(r'$\|\mu_m\|$ (rad/token)', fontsize=7, color=MUTED); top.tick_params(colors=MUTED, length=2)
        for spine in list(ax.spines.values()) + list(top.spines.values()):
            spine.set_visible(False)
        ax.set_xticks(np.arange(-.5, 16), minor=True); ax.set_yticks(np.arange(-.5, len(CLASSES)), minor=True)
        ax.grid(which='minor', color='white', linewidth=1.2); ax.tick_params(which='minor', length=0)
        bar = fig.colorbar(image, cax=bar_ax); bar.outline.set_visible(False); bar.ax.tick_params(labelsize=6.5)
        bar.set_label('Row-normalized mean atom attribution', fontsize=7)
        for suffix in ['pdf', 'png']:
            fig.savefig(OUT / f'class_atom_matrix_block{block}_large.{suffix}', bbox_inches='tight', dpi=300)
        plt.close(fig)

    write_json(OUT / 'provenance.json', dict(
        checkpoints={str(s): dict(path=p, sha256=results[s]['checkpoint_sha256']) for s, p in CHECKPOINTS.items()},
        sample=dict(images=len(frame), per_class_max=PER_CLASS, random_state=42,
                    correct_per_class={str(s): class_matrix(results[s]['values'][11], labels, results[s]['targets'])[1]
                                       for s in CHECKPOINTS}),
        dc_radius=DC_RADIUS, blocks=BLOCKS, post_hoc=True,
        source_sha256={p: sha256(ROOT / p) for p in ['experiments/round4_attribution.py', 'experiments/faithfulness.py']}))
    print(pd.DataFrame(matches).drop(columns='assignment').round(3).to_string(index=False))


if __name__ == '__main__':
    main()
