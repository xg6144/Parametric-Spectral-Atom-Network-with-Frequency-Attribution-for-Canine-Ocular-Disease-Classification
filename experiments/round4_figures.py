"""Round-4 figures: learning curves (30 and 100 epochs) and redesigned explanation examples.

The example panels use the archived example images (results/attribution/810d46669d84/examples.json)
and the validation-selected PSAN of seed 42. Each row shows the input with its true and predicted
class, the spatial Grad-CAM of the final block, the signed frequency attribution of block 11 (the
signed gradient-response product per stored frequency bin, which sums exactly to <grad f, Z>) and the
block-11 atom attributions with the atoms at the DC bin marked. Inference only.

    $PY -m experiments.round4_figures --device cuda:0
"""
from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from models.psan import _forward_from_block, _forward_to_block
from .common import CLASSES, ROOT, ManifestDataset, load_manifest, seed_all, write_json
from .faithfulness import attributions
from .round3_figures import DIVERGING, INK, MUTED, atom_parameters
from .round4_attribution import CHECKPOINTS, DC_RADIUS, load_psan

FIGURES = ROOT / 'results/round4/figures'
COLORS = {'psan': '#1f4e9c', 'psan_ad0': '#2a9d8f', 'gfnet_matched': '#c2410c', 'lowrank16': '#b8860b'}
STYLES = {'psan': '-', 'psan_ad0': '-.', 'gfnet_matched': '--', 'lowrank16': ':'}
NAMES = {'psan': 'PSAN (AD 0.2)', 'psan_ad0': 'PSAN (AD 0)', 'gfnet_matched': 'Dense', 'lowrank16': 'Free low-rank'}
LABELS = {c: c.replace('_', ' ').replace('Non ulcerative', 'Non-ulcerative') for c in CLASSES}
plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7,
                     'ytick.labelsize': 7, 'legend.fontsize': 7, 'pdf.fonttype': 42, 'ps.fonttype': 42,
                     'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED})


def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f'{name}.pdf', bbox_inches='tight')
    fig.savefig(FIGURES / f'{name}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def tidy(ax):
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.grid(axis='y', color='#e6e5e0', linewidth=.6)


def learning_curves():
    curves = pd.read_csv(ROOT / 'results/round4/tables/learning_curves.csv')
    short = curves[curves.budget == '30 epochs']
    long = curves[curves.budget == '100 epochs']
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.5), gridspec_kw=dict(width_ratios=[1, 1.45, 1.45]))
    arms = [a for a in ['psan', 'psan_ad0', 'gfnet_matched', 'lowrank16'] if a in set(short.arm)]
    for arm in arms:
        sub = short[short.arm == arm].pivot_table(index='epoch', columns='seed', values='validation_macro_f1')
        axes[0].plot(sub.index, sub.mean(axis=1), STYLES[arm], color=COLORS[arm], linewidth=1.4, label=NAMES[arm])
        axes[0].fill_between(sub.index, sub.min(axis=1), sub.max(axis=1), color=COLORS[arm], alpha=.12, linewidth=0)
    axes[0].set(title='30 epochs (seeds 42–44)', xlabel='Epoch', ylabel='Validation Macro F1', ylim=(.3, .72))
    for arm in [a for a in ['psan', 'psan_ad0', 'gfnet_matched'] if a in set(long.arm)]:
        sub = long[(long.arm == arm) & (long.seed == 42)].sort_values('epoch')
        axes[1].plot(sub.epoch, sub.validation_macro_f1, STYLES[arm], color=COLORS[arm], linewidth=1.4, label=NAMES[arm])
        axes[2].plot(sub.epoch, sub.train_loss, STYLES[arm], color=COLORS[arm], linewidth=1.4, label=NAMES[arm])
        best = sub.loc[sub.validation_macro_f1.idxmax()]
        axes[1].plot([best.epoch], [best.validation_macro_f1], 'o', color=COLORS[arm], markersize=4,
                     markeredgecolor='white', markeredgewidth=.8)
    axes[1].set(title='100 epochs (seed 42)', xlabel='Epoch', ylabel='Validation Macro F1', ylim=(.3, .72))
    axes[2].set(title='100 epochs (seed 42)', xlabel='Epoch', ylabel='Training loss')
    axes[2].set_yscale('log')
    for ax in axes:
        tidy(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(labels), frameon=False, bbox_to_anchor=(.5, 1.07))
    fig.tight_layout()
    save(fig, 'learning_curves')


def signed_frequency_map(model, x, target, index):
    """Signed gradient-response product per stored bin; sums to <grad f, Z> of the filter branch."""
    attr, c, h, logits, gradient = attributions(model, x, index, target)
    out = c.sum(1)
    spectrum = torch.fft.rfft2(out, dim=(1, 2), norm='ortho')
    grad_spec = torch.fft.rfft2(gradient, dim=(1, 2), norm='ortho')
    width = out.shape[2]
    multiplicity = torch.full((width // 2 + 1,), 2., device=out.device)
    multiplicity[0] = 1
    if width % 2 == 0:
        multiplicity[-1] = 1
    signed = (grad_spec.conj() * spectrum).real.sum(-1) * multiplicity[None, None, :]
    torch.testing.assert_close(signed.sum((1, 2)), (gradient * out).sum((1, 2, 3)), atol=2e-5, rtol=1e-4)
    return signed[0].cpu().numpy(), attr[0].cpu().numpy()


def gradcam(model, x, target):
    index = len(model.blocks) - 1
    with torch.no_grad():
        h = _forward_to_block(model, x, index)
        out = model.blocks[index].filter(model.blocks[index].norm1(h)); mixed = h + out
        features = (mixed + model.blocks[index].mlp(model.blocks[index].norm2(mixed))).detach()
    features.requires_grad_(True)
    logits = _forward_from_block(model, features, index + 1)
    grad = torch.autograd.grad(logits[0, target], features)[0]
    cam = torch.relu((features * grad.mean((1, 2), keepdim=True)).sum(-1)).detach()
    cam = torch.nn.functional.interpolate(cam[:, None], size=(224, 224), mode='bilinear', align_corners=False)[0, 0].cpu().numpy()
    return (cam - cam.min()) / (cam.max() - cam.min() + 1e-12)


def examples(device):
    seed_all(42)
    records = json.loads((ROOT / 'results/attribution/810d46669d84/examples.json').read_text())
    model = load_psan(CHECKPOINTS[42], device)
    radius = atom_parameters(load_psan(CHECKPOINTS[42], 'cpu'), 11)['radius']
    dc = radius < DC_RADIUS
    frame = load_manifest().set_index('path')
    provenance = []
    for cls in sorted({r['true_class'] for r in records}):
        chosen = sorted([r for r in records if r['true_class'] == cls], key=lambda r: not r['correct'])
        sub = frame.loc[[r['path'] for r in chosen]].reset_index()
        dataset = ManifestDataset(sub)
        fig = plt.figure(figsize=(7.16, 2.15 * len(chosen)))
        grid = fig.add_gridspec(len(chosen), 8, width_ratios=[1.15, .12, 1.15, .36, 1.0, .055, .9, 1.25], wspace=.05, hspace=.62)
        for row, record in enumerate(chosen):
            x, y = dataset[row]
            raw = (x.permute(1, 2, 0).numpy() * np.array([.229, .224, .225]) + np.array([.485, .456, .406])).clip(0, 1)
            x = x[None].to(device)
            with torch.no_grad():
                probabilities = model(x).softmax(-1)[0].cpu().numpy()
            target = int(probabilities.argmax())
            if CLASSES[target] != record['predicted_class']:
                raise ValueError('Example prediction changed for ' + record['path'])
            cam = gradcam(model, x, target)
            signed, values = signed_frequency_map(model, x, torch.tensor([target], device=device), 11)
            ax = fig.add_subplot(grid[row, 0]); ax.imshow(raw); ax.axis('off')
            verdict = 'Correct' if record['correct'] else 'Incorrect'
            ax.set_title(f'{verdict} (p = {probabilities[target]:.3f})\nTrue: {LABELS[record["true_class"]]}\n'
                         f'Predicted: {LABELS[CLASSES[target]]}', fontsize=6.5, loc='left')
            ax = fig.add_subplot(grid[row, 2]); ax.imshow(raw); ax.imshow(cam, cmap='inferno', alpha=.45, vmin=0, vmax=1)
            ax.axis('off'); ax.text(0, -.04, 'Grad-CAM (final block)', transform=ax.transAxes, fontsize=6.5, va='top', ha='left')
            ax = fig.add_subplot(grid[row, 4])
            shifted = np.fft.fftshift(signed, axes=0)
            limit = float(np.abs(shifted).max()) or 1.
            image = ax.imshow(shifted, cmap=DIVERGING, vmin=-limit, vmax=limit, aspect='auto', interpolation='nearest')
            ax.set_title('Signed frequency\nattribution (block 11)', fontsize=6.5, loc='left')
            ax.set_xlabel('Width freq. bin', fontsize=6); ax.set_ylabel('Height freq. bin', fontsize=6, labelpad=1)
            ax.set_yticks([0, 7, 13], ['$-7$', '0', '6']); ax.set_xticks([0, 7], ['0', '7']); ax.tick_params(labelsize=6)
            cax = fig.add_subplot(grid[row, 5])
            bar = fig.colorbar(image, cax=cax, ticks=[-limit, 0, limit]); bar.outline.set_visible(False)
            bar.ax.set_yticklabels([f'{-limit:.2g}', '0', f'{limit:.2g}']); bar.ax.tick_params(labelsize=5.5, length=2)
            ax = fig.add_subplot(grid[row, 7])
            colors = np.where(values >= 0, '#b52d2c', '#184f95')
            bars = ax.bar(range(16), values, color=colors, width=.75)
            for m in np.flatnonzero(dc):
                bars[m].set_hatch('////'); bars[m].set_edgecolor('white')
            ax.axhline(0, color=MUTED, linewidth=.6)
            ax.set_title('Atom attribution (block 11)\nhatched: atoms at the DC bin', fontsize=6.5, loc='left')
            ax.set_xlabel('Atom index $m$', fontsize=6); ax.set_xticks([0, 5, 10, 15]); ax.tick_params(labelsize=6); tidy(ax)
            provenance.append(dict(cls=cls, path=record['path'], correct=record['correct'], predicted=CLASSES[target],
                                   probability=float(probabilities[target])))
        save(fig, f'examples_{cls}')
    write_json(FIGURES / 'examples_provenance.json', dict(checkpoint=CHECKPOINTS[42], dc_atoms=[int(m) for m in np.flatnonzero(dc)],
                                                         examples=provenance))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--only', choices=['curves', 'examples'])
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.only != 'examples':
        learning_curves()
    if args.only != 'curves':
        examples(torch.device(args.device))


if __name__ == '__main__':
    main()
