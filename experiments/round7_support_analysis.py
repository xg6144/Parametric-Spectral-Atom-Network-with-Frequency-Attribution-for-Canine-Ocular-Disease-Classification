"""All-cell statistics for the frozen round7 unique-support intervention."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .common import ROOT, sha256, write_json
from .round3_analysis import Evaluator, holm, read_predictions, tail
from .round7_support import ARMS, DATASETS, FRACTIONS, OUT, PROTOCOL, SEEDS


def crossed_values(distributions, picks):
    """Use common seed picks and bootstrap groups for every simple/interaction contrast."""
    values = np.stack(distributions)
    return values[picks, np.arange(values.shape[1])[:, None]].mean(1)


def summarize(estimates, distributions, picks):
    values = crossed_values(distributions, picks)
    return dict(mean_delta=float(np.mean(estimates)), seed_sd=float(np.std(estimates, ddof=1)),
                ci_low=float(np.quantile(values, .025)), ci_high=float(np.quantile(values, .975)),
                p_uncorrected=tail(values), seeds_positive=int(np.count_nonzero(np.asarray(estimates) > 0)),
                seeds=','.join(map(str, SEEDS)), bootstrap=len(values))


def plot_contrasts(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    frame = pd.DataFrame(rows)
    plt.rcParams.update({'font.family': 'serif', 'font.size': 9, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.8), sharey=True, constrained_layout=True)
    labels = {'bloodmnist128': 'BloodMNIST', 'dermamnist_c128': 'Corrected DermaMNIST'}
    for ax, dataset in zip(axes, DATASETS):
        ax.axhline(0, color='.5', linestyle='--', linewidth=.8)
        for control, label, color, shift in [('gfnet_matched', 'PSAN − dense', '#0072B2', -.035),
                                              ('lowrank16', 'PSAN − free rank-16', '#D55E00', .035)]:
            values = frame[(frame.dataset == dataset) & (frame.control == control)].sort_values('fraction')
            x = np.arange(3, dtype=float) + shift
            mean = values.mean_delta.to_numpy() * 100
            low = values.ci_low.to_numpy() * 100; high = values.ci_high.to_numpy() * 100
            ax.errorbar(x, mean, yerr=np.vstack([mean - low, high - mean]), fmt='o-', color=color,
                        markersize=4, linewidth=1, capsize=3, label=label)
        ax.set_xticks(range(3), ['10%', '25%', '100%'])
        ax.set_xlabel('Nominal distinct training-group support')
        ax.set_title(labels[dataset]); ax.grid(axis='y', color='.9', linewidth=.5)
    axes[0].set_ylabel('Macro F1 difference (percentage points)')
    axes[1].legend(frameon=False, fontsize=8)
    fig.savefig(OUT / 'support_contrasts.pdf')
    fig.savefig(OUT / 'support_contrasts.png', dpi=200)
    plt.close(fig)


def run():
    spec = json.loads(PROTOCOL.read_text())
    tables = OUT / 'tables'; tables.mkdir(parents=True, exist_ok=True)
    results, source_paths = {}, []
    for path in sorted((OUT / 'final').glob('*/summary.json')):
        item = json.loads(path.read_text())
        if item['smoke'] or item['epochs_completed'] != 30 or item['protocol_sha256'] != sha256(PROTOCOL):
            raise ValueError('Incomplete or mismatched support result: ' + str(path))
        if sha256(ROOT / item['checkpoint']) != item['checkpoint_sha256']:
            raise ValueError('Checkpoint changed: ' + str(path))
        key = (item['dataset'], item['arm'], item['seed'], item['support_percentage'])
        if key in results: raise ValueError('Duplicate support cell.')
        results[key] = item
        source_paths.extend([path, path.with_name('test_predictions.csv')])
    expected = {(d, a, s, f) for d in DATASETS for a in ARMS for s in SEEDS for f in FRACTIONS}
    if set(results) != expected:
        raise ValueError(f'Need all54 cells; present={len(results)} missing={sorted(expected-set(results))}')
    run_rows, seed_rows, simple_rows, interaction_rows = [], [], [], []
    for dataset_index, dataset in enumerate(DATASETS):
        first = results[(dataset, 'psan_ad0', 42, 10)]
        path = (ROOT / first['checkpoint']).with_name('test_predictions.csv')
        reference, _, classes = read_predictions(path)
        b = spec['statistics']['bootstrap']
        evaluator = Evaluator(reference, b=b, seed=spec['statistics']['seed'] + dataset_index)
        rng = np.random.default_rng(spec['statistics']['seed'] + 100 + dataset_index)
        picks = rng.integers(0, len(SEEDS), (b, len(SEEDS)))
        scores = {}
        for arm in ARMS:
            for fraction in FRACTIONS:
                for seed in SEEDS:
                    item = results[(dataset, arm, seed, fraction)]
                    prediction_path = (ROOT / item['checkpoint']).with_name('test_predictions.csv')
                    frame, _, current_classes = read_predictions(prediction_path)
                    if current_classes != classes or not np.array_equal(frame.group_id, reference.group_id):
                        raise ValueError('Unpaired test labels/groups/classes.')
                    score, distribution = evaluator.score(prediction_path, len(classes))
                    if abs(score - item['test']['macro_f1']) > 1e-10:
                        raise ValueError('Saved test metric does not reproduce.')
                    scores[(arm, seed, fraction)] = (score, distribution)
                    run_rows.append(dict(dataset=dataset, arm=arm, seed=seed, fraction=fraction,
                                         macro_f1=score, validation_macro_f1=item['best_validation_macro_f1'],
                                         best_epoch=item['best_validation_epoch'], epochs=item['epochs_completed'],
                                         support_images=item['support_images'], support_groups=item['support_groups'],
                                         full_training_images=item['full_training_images'],
                                         draws_per_epoch=item['sampler_draws_per_epoch']))
        simple, differences = [], {}
        for control in ['gfnet_matched', 'lowrank16']:
            for fraction in FRACTIONS:
                estimates, distributions = [], []
                for seed in SEEDS:
                    oa, da = scores[('psan_ad0', seed, fraction)]
                    ob, db = scores[(control, seed, fraction)]
                    estimate, distribution = oa - ob, da - db
                    estimates.append(estimate); distributions.append(distribution)
                    differences[(control, seed, fraction)] = (estimate, distribution)
                    seed_rows.append(dict(dataset=dataset, control=control, fraction=fraction, seed=seed,
                                          delta=estimate, ci_low=float(np.quantile(distribution, .025)),
                                          ci_high=float(np.quantile(distribution, .975)),
                                          p_uncorrected=tail(distribution)))
                simple.append(dict(dataset=dataset, control=control, fraction=fraction,
                                   **summarize(estimates, distributions, picks)))
        for row, adjusted in zip(simple, holm([r['p_uncorrected'] for r in simple])):
            row.update(p_holm=float(adjusted), family_size=6)
        simple_rows.extend(simple)
        interactions = []
        for control in ['gfnet_matched', 'lowrank16']:
            for fraction in [10, 25]:
                estimates, distributions = [], []
                for seed in SEEDS:
                    oa, da = differences[(control, seed, fraction)]
                    ob, db = differences[(control, seed, 100)]
                    estimates.append(oa - ob); distributions.append(da - db)
                interactions.append(dict(dataset=dataset, control=control, fraction=fraction,
                                         reference_fraction=100,
                                         definition='(PSAN-control) at reduced support minus (PSAN-control) at full support',
                                         **summarize(estimates, distributions, picks)))
        for row, adjusted in zip(interactions, holm([r['p_uncorrected'] for r in interactions])):
            row.update(p_holm=float(adjusted), family_size=4)
        interaction_rows.extend(interactions)
        print(dataset, 'all27 cells analyzed', flush=True)
    pd.DataFrame(run_rows).to_csv(tables / 'per_run.csv', index=False)
    pd.DataFrame(seed_rows).to_csv(tables / 'per_seed_contrasts.csv', index=False)
    pd.DataFrame(simple_rows).to_csv(tables / 'contrasts.csv', index=False)
    pd.DataFrame(interaction_rows).to_csv(tables / 'interactions.csv', index=False)
    aggregate = pd.DataFrame(run_rows).groupby(['dataset', 'fraction', 'arm']).macro_f1.agg(['mean', 'std']).reset_index()
    aggregate.to_csv(tables / 'aggregate.csv', index=False)
    plot_contrasts(simple_rows)
    write_json(OUT / 'analysis_provenance.json', dict(
        protocol_sha256=sha256(PROTOCOL), analysis_source_sha256=sha256(ROOT / 'experiments/round7_support_analysis.py'),
        source_sha256={str(p.relative_to(ROOT)): sha256(p) for p in source_paths}, completed_cells=54,
        group_uncertainty='Test lesions for corrected Derma and images for Blood; crossed with only3 empirical seeds that jointly vary training support and initialization/optimization.',
        intervals='Unadjusted95% percentile intervals; p values Holm-adjusted separately for6 simple and4 interaction contrasts per domain.',
        stage='Disclosed post-hoc exploration specified before these new runs; not independent blinded confirmation.'))
    print(pd.DataFrame(simple_rows).to_string(index=False), flush=True)
    print(pd.DataFrame(interaction_rows).to_string(index=False), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=4): run()
