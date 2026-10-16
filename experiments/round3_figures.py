"""Round-3 figures from existing checkpoints/attribution records (no training, no new inference).

Class-atom attribution matrix rebuilt from the archived 220-image diagnostic records of the
validation-selected PSAN (seed 42), using correctly classified images so that the explained logit is
the true class, and the learned atom envelopes of the same checkpoint.

    $PY -m experiments.round3_figures
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from .common import CLASSES, ROOT, load_model, sha256, write_json

FIGURES = ROOT / 'results/round3/figures'
CHECKPOINT = ROOT / 'results/tuning/final/canine_psan_c2_s42_final/best.pth'
ATTRIBUTION = ROOT / 'results/attribution/810d46669d84/atom_values.json'
DIVERGING = LinearSegmentedColormap.from_list('blue_gray_red', ['#184f95', '#6da7ec', '#f0efec', '#ec8c8b', '#b52d2c'])
SEQUENTIAL = LinearSegmentedColormap.from_list('blue_seq', ['#fcfcfb', '#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b'])
INK, MUTED, FAINT = '#0b0b0b', '#52514e', '#c3c2b7'
LABELS = {c: c.replace('_', ' ').replace('Non ulcerative', 'Non-ulcerative') for c in CLASSES}
plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7,
                     'ytick.labelsize': 7, 'pdf.fonttype': 42, 'ps.fonttype': 42,
                     'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED})


def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f'{name}.pdf', bbox_inches='tight')
    fig.savefig(FIGURES / f'{name}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def atom_parameters(model, block):
    f = model.blocks[block].filter
    p = {name: getattr(f, name).detach().double().numpy() for name in ['mu', 'log_sigma', 'rho_raw', 'log_amp', 'w_real', 'w_imag']}
    p.update(theta=f.theta.detach().double().numpy(), phi=f.phi.detach().double().numpy(),
             grid_u=f.U.double().numpy(), grid_v=f.V.double().numpy(), atom=f.cfg.atom, M=f.cfg.M,
             anisotropic=f.cfg.anisotropic)
    p['radius'] = np.linalg.norm(p['mu'], axis=1)
    p['mixing_norm'] = np.sqrt((p['w_real'] ** 2 + p['w_imag'] ** 2).sum(0))
    return p


def envelope(p, u, v):
    """Exact PSANFilter._envelope formula evaluated on arbitrary (u, v) coordinates."""
    du = u[None] - p['mu'][:, 0, None, None]
    dv = v[None] - p['mu'][:, 1, None, None]
    s = np.exp(p['log_sigma'])
    if p['anisotropic']:
        rho = np.tanh(p['rho_raw']) * .95
        off = rho * s[:, 0] * s[:, 1]
        det = np.maximum(s[:, 0] ** 2 * s[:, 1] ** 2 - off ** 2, 1e-8)
        a, b, c = (s[:, 1] ** 2 / det)[:, None, None], (s[:, 0] ** 2 / det)[:, None, None], (-off / det)[:, None, None]
        quad = a * du ** 2 + b * dv ** 2 + 2 * c * du * dv
    else:
        quad = du ** 2 / (s[:, 0] ** 2)[:, None, None] + dv ** 2 / (s[:, 1] ** 2)[:, None, None]
    return np.exp(p['log_amp'])[:, None, None] * np.exp(-.5 * quad)


def class_atom_matrix(block):
    records = [r for r in json.loads(ATTRIBUTION.read_text()) if r['block'] == block]
    rows, counts = [], []
    for k in range(len(CLASSES)):
        chosen = [np.asarray(r['attribution']) for r in records if r['true_label'] == k and r['target'] == k]
        counts.append(len(chosen))
        mean = np.mean(chosen, 0) if chosen else np.zeros(16)
        rows.append(mean / max(np.linalg.norm(mean), 1e-12))
    return np.asarray(rows), counts, len(records)


def figure_matrix(p, block):
    matrix, counts, records = class_atom_matrix(block)
    order = np.argsort(p['radius'])
    limit = np.ceil(np.abs(matrix).max() * 10) / 10
    fig = plt.figure(figsize=(7.16, 2.9))
    ax = fig.add_axes([.24, .12, .62, .70]); bar_ax = fig.add_axes([.875, .12, .014, .70])
    image = ax.imshow(matrix[:, order], cmap=DIVERGING, vmin=-limit, vmax=limit, aspect='auto', interpolation='nearest')
    ax.set_xticks(range(len(order)), [str(m) for m in order])
    ax.set_yticks(range(len(CLASSES)), [f'{LABELS[c]} ({n})' for c, n in zip(CLASSES, counts)])
    ax.set_xlabel('Atom index $m$ (block %d), ordered by learned center radius' % block)
    top = ax.secondary_xaxis('top')
    top.set_xticks(range(len(order)), [f'{p["radius"][m]:.2f}' for m in order], fontsize=6)
    top.set_xlabel(r'$\|\mu_m\|$ (rad/token)', fontsize=7, color=MUTED); top.tick_params(colors=MUTED, length=2)
    for spine in list(ax.spines.values()) + list(top.spines.values()):
        spine.set_visible(False)
    ax.set_xticks(np.arange(-.5, len(order)), minor=True); ax.set_yticks(np.arange(-.5, len(CLASSES)), minor=True)
    ax.grid(which='minor', color='white', linewidth=1.2); ax.tick_params(which='minor', length=0)
    bar = fig.colorbar(image, cax=bar_ax); bar.outline.set_visible(False); bar.ax.tick_params(labelsize=6.5)
    bar.set_label('Row-normalized mean atom attribution', fontsize=7)
    save(fig, f'class_atom_matrix_block{block}')
    return dict(block=block, diagnostic_records=records, correct_per_class=dict(zip(CLASSES, counts)),
                atom_order_by_radius=[int(m) for m in order], color_limit=float(limit),
                matrix=[[float(x) for x in row] for row in matrix])


def figure_envelopes(p, block):
    order = np.argsort(p['radius'])
    v = np.linspace(0, np.pi, 160); u = np.linspace(-np.pi, np.pi, 320)
    grid_v, grid_u = np.meshgrid(v, u)
    env = envelope(p, grid_u, grid_v)
    stored = envelope(p, p['grid_u'], p['grid_v'])
    fig, axes = plt.subplots(2, 8, figsize=(7.16, 3.3), sharex=True, sharey=True)
    for ax, m in zip(axes.ravel(), order):
        ax.imshow(env[m] / env[m].max(), origin='lower', extent=[0, np.pi, -np.pi, np.pi], cmap=SEQUENTIAL,
                  vmin=0, vmax=1, aspect='auto', interpolation='bilinear')
        ax.scatter(p['grid_v'].ravel(), p['grid_u'].ravel(), s=1.2, color=FAINT, linewidths=0)
        ax.set_title(f'$m$={m}\n$\\|\\mu\\|$={p["radius"][m]:.2f}', fontsize=6.2, color=INK, pad=2, linespacing=1.1)
        ax.text(.04, .03, f'$A\\|w\\|$={np.exp(p["log_amp"][m]) * p["mixing_norm"][m]:.2f}', transform=ax.transAxes,
                fontsize=5.8, color=INK)
        ax.set_xticks([0, np.pi], ['0', r'$\pi$']); ax.set_yticks([-np.pi, 0, np.pi], [r'$-\pi$', '0', r'$\pi$'])
        ax.tick_params(length=2, labelsize=6)
        for spine in ax.spines.values():
            spine.set_color(FAINT)
    fig.supxlabel(r'$v$ (rad/token), stored non-negative half-plane', fontsize=7)
    fig.supylabel(r'$u$ (rad/token)', fontsize=7)
    fig.subplots_adjust(wspace=.12, hspace=.42)
    save(fig, f'atom_envelopes_block{block}')
    return dict(block=block, atom_order_by_radius=[int(m) for m in order],
                radius=[float(x) for x in p['radius']], mixing_norm=[float(x) for x in p['mixing_norm']],
                amplitude=[float(x) for x in np.exp(p['log_amp'])],
                stored_bin_mass_fraction=[float(stored[m].sum() / max(env[m].sum() * (stored[m].size / env[m].size), 1e-12))
                                          for m in range(p['M'])])


SERIES = {'psan': ('PSAN', '#2a78d6', 'o', '-'), 'gfnet': ('GFNet (repo)', '#eb6834', 's', '--'),
          'afno': ('AFNO (repo)', '#1baf7a', '^', ':')}
CODES = ['BL', 'CA', 'CJ', 'EN', 'EP', 'ET', 'NK', 'NO', 'NS', 'PK', 'UK']


def predictions(path):
    import pandas as pd
    frame = pd.read_csv(path, keep_default_na=False)
    logits = frame.filter(regex=r'^logit::').to_numpy(dtype=float)
    return frame.label.to_numpy(int), logits


def figure_confusion():
    runs = [('PSAN (AD 0.2)', ROOT / 'results/training/canine_psan_gabor_ad0p2_M16_morlet_s42/test_predictions.csv'),
            ('Matched dense filter', ROOT / 'results/training/canine_gfnet_matched_s42/test_predictions.csv')]
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.55), constrained_layout=True)
    out = {}
    for ax, (title, path) in zip(axes, runs):
        y, logits = predictions(path)
        k = len(CLASSES)
        counts = np.bincount(y * k + logits.argmax(1), minlength=k * k).reshape(k, k).astype(float)
        rates = counts / counts.sum(1, keepdims=True)
        image = ax.imshow(rates, cmap=SEQUENTIAL, vmin=0, vmax=1)
        for i in range(k):
            for j in range(k):
                if rates[i, j] >= .01:
                    ax.text(j, i, f'{rates[i, j]:.2f}'.lstrip('0'), ha='center', va='center', fontsize=5.6,
                            color='white' if rates[i, j] > .55 else INK)
        ax.set_xticks(range(k), CODES, fontsize=6.5); ax.set_yticks(range(k), CODES, fontsize=6.5)
        ax.set_xlabel('Predicted class'); ax.set_ylabel('True class')
        ax.set_title(title, color=INK)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)
        out[title] = dict(path=str(path.relative_to(ROOT)), sha256=sha256(path), recall=[float(x) for x in np.diag(rates)])
    bar = fig.colorbar(image, ax=axes, shrink=.8, pad=.01); bar.outline.set_visible(False)
    bar.set_label('Row-normalized proportion', fontsize=7); bar.ax.tick_params(labelsize=6.5)
    save(fig, 'confusion_psan_dense_s42')
    return out


def reliability_curve(probabilities, y, bins=15):
    confidence = probabilities.max(1); correct = probabilities.argmax(1) == y
    edges = np.linspace(0, 1, bins + 1)
    index = np.clip(np.digitize(confidence, edges[1:-1]), 0, bins - 1)
    rows, ece = [], 0.
    for b in range(bins):
        mask = index == b
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
            rows.append((confidence[mask].mean(), correct[mask].mean(), int(mask.sum())))
    return np.array(rows), float(ece), confidence


def figure_reliability(label, prediction_path, temperature, name):
    y, logits = predictions(prediction_path)
    softmax = lambda z: np.exp(z - z.max(1, keepdims=True)) / np.exp(z - z.max(1, keepdims=True)).sum(1, keepdims=True)
    raw_rows, raw_ece, _ = reliability_curve(softmax(logits), y)
    scaled_rows, scaled_ece, scaled_conf = reliability_curve(softmax(logits / temperature), y)
    fig, (ax, hist) = plt.subplots(2, 1, figsize=(3.45, 3.55), sharex=True, gridspec_kw=dict(height_ratios=[3, 1], hspace=.08))
    ax.plot([0, 1], [0, 1], color=FAINT, lw=1, zorder=1)
    keep = lambda rows: rows[rows[:, 2] >= 20]
    r, q = keep(raw_rows), keep(scaled_rows)
    ax.plot(r[:, 0], r[:, 1], color='#8a8986', ls='--', marker='o', ms=3.2, lw=1.2, label=f'Raw (ECE {raw_ece:.3f})')
    ax.plot(q[:, 0], q[:, 1], color='#2a78d6', ls='-', marker='s', ms=3.2, lw=1.4, label=f'Temperature $T$={temperature:.2f} (ECE {scaled_ece:.3f})')
    ax.set_ylabel('Accuracy'); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(frameon=False, fontsize=6.5, loc='upper left')
    ax.grid(color='#e9e8e4', lw=.6); ax.set_title(label, color=INK, fontsize=7.5)
    hist.hist(scaled_conf, bins=np.linspace(0, 1, 16), color='#86b6ef', edgecolor='white', linewidth=.6)
    hist.set_xlabel('Confidence'); hist.set_ylabel('Images'); hist.tick_params(labelsize=6.5)
    for a in (ax, hist):
        for side in ['top', 'right']:
            a.spines[side].set_visible(False)
    save(fig, name)
    return dict(path=str(prediction_path.relative_to(ROOT)), temperature=temperature, raw_ece=raw_ece, scaled_ece=scaled_ece)


def figure_robustness():
    import pandas as pd
    path = ROOT / 'results/tables/robustness.csv'
    frame = pd.read_csv(path)
    corruptions = [('gaussian_noise', 'Gaussian noise'), ('gaussian_blur', 'Gaussian blur'), ('jpeg_quality', 'JPEG compression'),
                   ('brightness_factor', 'Brightness'), ('contrast_factor', 'Contrast'), ('saturation_factor', 'Saturation')]
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 3.9), sharex=True)
    for ax, (key, title) in zip(axes.ravel(), corruptions):
        for model, (label, color, marker, style) in SERIES.items():
            clean = frame[(frame.model == model) & (frame.corruption == 'clean')].macro_f1.iloc[0]
            sub = frame[(frame.model == model) & (frame.corruption == key)].sort_values('severity')
            ax.plot([0] + sub.severity.tolist(), [clean] + sub.macro_f1.tolist(), color=color, marker=marker, ls=style,
                    ms=3.5, lw=1.4, label=label)
        ax.set_title(title, color=INK); ax.set_xticks([0, 1, 2, 3], ['clean', '1', '2', '3'])
        ax.grid(color='#e9e8e4', lw=.6)
        for side in ['top', 'right']:
            ax.spines[side].set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel('Test Macro F1')
    for ax in axes[1]:
        ax.set_xlabel('Severity')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False, fontsize=7.5, bbox_to_anchor=(.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, .95))
    save(fig, 'robustness')
    return dict(source=str(path.relative_to(ROOT)), sha256=sha256(path))


def run():
    model, _, _ = load_model(CHECKPOINT, 'cpu')
    provenance = dict(checkpoint=str(CHECKPOINT.relative_to(ROOT)), checkpoint_sha256=sha256(CHECKPOINT),
                      attribution=str(ATTRIBUTION.relative_to(ROOT)), attribution_sha256=sha256(ATTRIBUTION),
                      selection='correctly classified diagnostic images (target == true label); rows L2-normalized',
                      caveat='atom indices are run-specific and not identifiable across trainings; descriptive only',
                      figures={})
    for block in [11, 5, 0]:
        p = atom_parameters(model, block)
        provenance['figures'][f'class_atom_matrix_block{block}'] = figure_matrix(p, block)
        provenance['figures'][f'atom_envelopes_block{block}'] = figure_envelopes(p, block)
    import pandas as pd
    calibration = pd.read_csv(ROOT / 'results/tables/calibration.csv', keep_default_na=False)
    temperature = lambda model: float(calibration[(calibration.model == model) & (calibration.condition == 'temperature')].temperature.iloc[0])
    provenance['figures']['confusion_psan_dense_s42'] = figure_confusion()
    provenance['figures']['reliability_tuned_psan'] = figure_reliability(
        'Validation-selected PSAN (seed 42)', ROOT / 'results/tuning/final/canine_psan_c2_s42_final/test_predictions.csv',
        temperature('canine_psan_c2_s42_final'), 'reliability_tuned_psan')
    provenance['figures']['reliability_common_psan'] = figure_reliability(
        'Shared-preset PSAN (seed 42)', ROOT / 'results/training/canine_psan_gabor_ad0p2_M16_morlet_s42/test_predictions.csv',
        temperature('canine_psan_gabor_ad0p2_M16_morlet_s42'), 'reliability_common_psan')
    provenance['figures']['robustness'] = figure_robustness()
    write_json(FIGURES / 'figures_provenance.json', provenance)
    print('Round-3 figures written to', FIGURES, flush=True)


if __name__ == '__main__':
    run()
