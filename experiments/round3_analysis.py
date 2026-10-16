"""Round-3 statistics and audit tables from saved predictions (no model training).

Paired percentile bootstrap with B=10,000 resamples of evaluation groups (visual groups for the
canine test set, lesions for corrected Derma, images otherwise). Across-seed summaries use a
crossed bootstrap of groups and seed indices. The two-sided tail probability is the smoothed
(count+1)/(B+1) estimate, so the smallest attainable value is 2/(B+1). Round-2 families keep their
declared Bonferroni sizes; round-3 families use Holm. Outputs: results/round3/tables/*.csv.

    $PY -m experiments.round3_analysis
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import CLASSES, ROOT, sha256, write_json

OUT = ROOT / 'results/round3'
TABLES = OUT / 'tables'
B = 10000
SEED = 2026
NORMAL = CLASSES.index('Normal')
DISEASES = [i for i in range(len(CLASSES)) if i != NORMAL]
PRETRAINED = ['convnext', 'deit', 'fastvit_t8', 'shvit_s1', 'swin']


# ----------------------------------------------------------------------------- loading

def read_predictions(path):
    frame = pd.read_csv(path, keep_default_na=False)
    logits = frame.filter(regex=r'^logit::').to_numpy(dtype=np.float64)
    names = [c.split('::', 1)[1] for c in frame.columns if c.startswith('logit::')]
    if frame.path.duplicated().any():
        raise ValueError('Duplicate prediction rows in ' + str(path))
    return frame[['path', 'label', 'group_id']].copy(), logits, names


def summary_predictions(summary):
    return (ROOT / summary['checkpoint']).with_name('test_predictions.csv')


def common_run(tag):
    path = ROOT / 'results/training' / tag / 'summary.json'
    return json.loads(path.read_text()) if path.exists() else None


def canine_runs():
    """Common-preset canine runs keyed by (arm, seed)."""
    arms = {'psan': 'canine_psan_gabor_ad0p2_M16_morlet_s{}', 'psan_ad0': 'canine_psan_gabor_ad0p0_M16_morlet_s{}',
            'psan_gauss_ad0': 'canine_psan_gaussian_ad0p0_M16_morlet_s{}',
            'psan_gauss_ad02': 'canine_psan_gaussian_ad0p2_M16_morlet_s{}',
            'gfnet': 'canine_gfnet_s{}', 'afno': 'canine_afno_s{}', 'gfnet_matched': 'canine_gfnet_matched_s{}'}
    runs = {}
    for arm, pattern in arms.items():
        for seed in range(42, 47):
            s = common_run(pattern.format(seed))
            if s:
                runs[(arm, seed)] = s
    for model in PRETRAINED:
        s = common_run(f'canine_{model}_imagenet_s42')
        if s:
            runs[(model, 42)] = s
    for atoms in [4, 8, 32, 64]:
        for seed in range(42, 47):
            s = common_run(f'canine_psan_gabor_ad0p0_M{atoms}_morlet_s{seed}')
            r3 = OUT / 'training' / f'canine_psan_gabor_ad0p0_M{atoms}_morlet_s{seed}' / 'summary.json'
            if s is None and r3.exists():
                s = json.loads(r3.read_text())
            if s:
                runs[(f'psan_ad0_M{atoms}', seed)] = s
    for variant in ['canine_psan_gabor_ad0p0_M16_morlet_axis_s42', 'canine_psan_gabor_ad0p0_M16_morlet_fixedphase_s42',
                    'canine_psan_gabor_ad0p0_M16_random_s42', 'canine_psan_gabor_ad0p1_M16_morlet_s42']:
        s = common_run(variant)
        if s:
            runs[(variant, 42)] = s
    return runs


def tuned_runs():
    """Validation-selected canine runs keyed by (arm, seed): round-2 seed 42 plus round-3 runs."""
    runs = {}
    for path in (ROOT / 'results/tuning/final_selected').glob('*.json'):
        runs[(path.stem, 42)] = json.loads(path.read_text())
    for path in (OUT / 'tuning/final').glob('*/summary.json'):
        s = json.loads(path.read_text())
        runs[(s['arm'], s['seed'])] = s
    return runs


def external_runs():
    runs = {}
    for path in (OUT / 'external/final').glob('*/summary.json'):
        s = json.loads(path.read_text())
        runs[(s['dataset'], s['arm'], s['seed'])] = s
    return runs


# ----------------------------------------------------------------------------- bootstrap

class GroupBootstrap:
    """Shared multinomial resampling of evaluation groups for paired comparisons."""

    def __init__(self, groups, b=B, seed=SEED):
        _, self.codes = np.unique(np.asarray(groups), return_inverse=True)
        self.groups = int(self.codes.max() + 1)
        rng = np.random.default_rng(seed)
        self.counts = np.empty((b, self.groups), dtype=np.float32)
        for i in range(b):
            self.counts[i] = np.bincount(rng.integers(0, self.groups, self.groups), minlength=self.groups)
        self.b = b

    def macro_f1(self, y, pred, k, labels=None):
        """Observed and resampled Macro F1 (all-class confusion, absent classes contribute 0)."""
        index = y * k + pred
        per_group = np.zeros((self.groups, k * k), dtype=np.float32)
        np.add.at(per_group, (self.codes, index), 1.)
        def score(confusion):
            confusion = confusion.reshape(-1, k, k)
            tp = np.diagonal(confusion, axis1=1, axis2=2)
            denominator = confusion.sum(1) + confusion.sum(2)
            f1 = np.divide(2 * tp, denominator, out=np.zeros_like(tp), where=denominator > 0)
            return (f1 if labels is None else f1[:, labels]).mean(1)
        observed = float(score(per_group.sum(0, keepdims=True).astype(np.float64))[0])
        return observed, score(self.counts @ per_group)


def observed_f1(y, pred, k, labels=None):
    confusion = np.bincount(y * k + pred, minlength=k * k).reshape(k, k).astype(float)
    tp = np.diag(confusion); denominator = confusion.sum(0) + confusion.sum(1)
    f1 = np.divide(2 * tp, denominator, out=np.zeros(k), where=denominator > 0)
    return float((f1 if labels is None else f1[labels]).mean())


def tail(distribution):
    n = len(distribution)
    return min(1., 2 * min((np.count_nonzero(distribution <= 0) + 1) / (n + 1),
                          (np.count_nonzero(distribution >= 0) + 1) / (n + 1)))


def holm(pvalues):
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adjusted[i] = min(1., running)
    return adjusted


class Evaluator:
    """Aligned predictions on one evaluation set, with a shared bootstrap."""

    def __init__(self, reference_frame, groups=None, b=B, seed=SEED, mask=None):
        self.frame = reference_frame.reset_index(drop=True)
        self.mask = np.ones(len(self.frame), bool) if mask is None else mask
        self.y = self.frame.label.to_numpy(int)[self.mask]
        group_values = (self.frame.group_id if groups is None else groups).to_numpy()[self.mask]
        self.boot = GroupBootstrap(group_values, b, seed)
        self.cache = {}

    def score(self, path, k, labels=None):
        cache_key = (str(path), None if labels is None else tuple(labels))
        if cache_key not in self.cache:
            frame, logits, _ = read_predictions(path)
            if not (np.array_equal(frame.path, self.frame.path) and np.array_equal(frame.label, self.frame.label)):
                raise ValueError('Unpaired predictions: ' + str(path))
            pred = logits.argmax(1)[self.mask]
            self.cache[cache_key] = self.boot.macro_f1(self.y, pred, k, labels)
        return self.cache[cache_key]


def contrast_rows(evaluator, pairs, k, labels=None, family=None, adjust='bonferroni', label=''):
    """pairs: list of (comparison, seed, path_a, path_b). Returns per-seed paired rows."""
    rows = []
    for comparison, seed, path_a, path_b in pairs:
        oa, da = evaluator.score(path_a, k, labels)
        ob, db = evaluator.score(path_b, k, labels)
        difference = da - db
        rows.append(dict(analysis=label, comparison=comparison, seed=seed, f1_a=oa, f1_b=ob, delta_macro_f1=oa - ob,
                         ci_low=float(np.quantile(difference, .025)), ci_high=float(np.quantile(difference, .975)),
                         p_uncorrected=tail(difference), _dist=difference))
    if rows:
        p = [r['p_uncorrected'] for r in rows]
        size = family or len(rows)
        adjusted = holm(p) if adjust == 'holm' else np.minimum(1., np.asarray(p) * size)
        for r, value in zip(rows, adjusted):
            r.update(p_adjusted=float(value), adjustment=adjust, family_size=size)
    return rows


def crossed(rows, comparison, seed=SEED + 7):
    """Crossed group x seed bootstrap of the mean paired difference across the seeds in rows."""
    selected = [r for r in rows if r['comparison'] == comparison]
    if len(selected) < 2:
        return None
    distributions = np.stack([r['_dist'] for r in selected])
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(selected), (distributions.shape[1], len(selected)))
    values = distributions[picks, np.arange(distributions.shape[1])[:, None]].mean(1)
    deltas = np.array([r['delta_macro_f1'] for r in selected])
    sd = deltas.std(ddof=1)
    return dict(comparison=comparison, seeds=','.join(str(r['seed']) for r in selected), runs=len(selected),
                mean_delta=float(deltas.mean()), seed_sd=float(sd),
                crossed_ci_low=float(np.quantile(values, .025)), crossed_ci_high=float(np.quantile(values, .975)),
                p_crossed=tail(values), seed_level_t=float(deltas.mean() / (sd / np.sqrt(len(deltas)))) if sd > 0 else None,
                seeds_positive=int((deltas > 0).sum()))


def strip(rows):
    return [{k: v for k, v in r.items() if not k.startswith('_')} for r in rows]


# ----------------------------------------------------------------------------- analyses

def audit_tables():
    manifest = pd.read_csv(ROOT / 'results/corrected_dedup_manifest.csv', usecols=['class', 'status', 'reason', 'duplicate_group'],
                           low_memory=False)
    split = pd.read_csv(ROOT / 'results/revision_split.csv', usecols=['class', 'split'], keep_default_na=False)
    table = pd.crosstab(manifest['class'], manifest['status'])
    table = table.rename(columns={'keep': 'retained', 'exclude_duplicate': 'duplicate_excluded',
                                  'exclude_label_conflict': 'conflict_excluded'})
    table['raw'] = table.sum(axis=1)
    table = table.join(pd.crosstab(split['class'], split['split']))
    table['retained_percent'] = 100 * table['retained'] / table['raw']
    table = table[['raw', 'duplicate_excluded', 'conflict_excluded', 'retained', 'retained_percent', 'train', 'val', 'test']]
    total = table.sum(numeric_only=True)
    total['retained_percent'] = 100 * total['retained'] / total['raw']
    table.loc['Total'] = total
    table.index.name = 'class'
    table.reset_index().to_csv(TABLES / 'audit_class_distribution.csv', index=False)
    conflicts = manifest[manifest.status == 'exclude_label_conflict']
    groups = conflicts.drop_duplicates('duplicate_group').copy()
    groups['class_set'] = groups.reason.str.replace('duplicate group spans classes: ', '', regex=False)
    images = conflicts.assign(class_set=conflicts.reason.str.replace('duplicate group spans classes: ', '', regex=False))
    sets = (groups.class_set.value_counts().rename('groups').to_frame()
            .join(images.class_set.value_counts().rename('images')))
    sets['classes_in_set'] = sets.index.str.count(r'\|') + 1
    sets['includes_normal'] = sets.index.str.contains('Normal')
    sets.index.name = 'class_set'
    sets.reset_index().to_csv(TABLES / 'audit_conflict_sets.csv', index=False)
    size = groups.class_set.str.count(r'\|') + 1
    pairs = {}
    for value in groups.class_set:
        members = value.split('|')
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                pairs[(members[i], members[j])] = pairs.get((members[i], members[j]), 0) + 1
    pair_frame = pd.DataFrame([dict(class_a=a, class_b=b, conflict_groups=n) for (a, b), n in pairs.items()])
    pair_frame.sort_values('conflict_groups', ascending=False).to_csv(TABLES / 'audit_conflict_pairs.csv', index=False)
    return dict(conflict_images=int(len(conflicts)), conflict_groups=int(len(groups)),
                groups_including_normal=int(groups.class_set.str.contains('Normal').sum()),
                classes_per_group=size.value_counts().sort_index().to_dict())


def canine_contrasts(runs, tuned):
    reference_path = summary_predictions(runs[('psan', 42)])
    reference, _, names = read_predictions(reference_path)
    if names != CLASSES:
        raise ValueError('Unexpected class order')
    evaluator = Evaluator(reference)
    path = lambda s: summary_predictions(s)
    outputs = {}

    # Round-2 primary family (18 comparisons, Bonferroni 18), recomputed with B=10,000.
    pairs = []
    for model, seeds in [('afno', range(42, 47)), ('gfnet', range(42, 47)), ('gfnet_matched', range(42, 45))] + \
                        [(m, [42]) for m in PRETRAINED]:
        for seed in seeds:
            if (model, seed) in runs:
                pairs.append(('PSAN minus ' + model, seed, path(runs[('psan', seed)]), path(runs[(model, seed)])))
    main = contrast_rows(evaluator, pairs, 11, family=18, label='round2 primary family recomputed')
    outputs['main_contrasts'] = main
    outputs['main_crossed'] = [c for c in (crossed(main, 'PSAN minus ' + m) for m in ['gfnet', 'afno', 'gfnet_matched']) if c]

    # Dense-filter controls under the common preset (post hoc; disclosed).
    pairs = []
    for arm in ['psan', 'psan_ad0', 'psan_gauss_ad0', 'psan_gauss_ad02']:
        for seed in range(42, 45):
            if (arm, seed) in runs and ('gfnet_matched', seed) in runs:
                pairs.append((arm + ' minus gfnet_matched', seed, path(runs[(arm, seed)]), path(runs[('gfnet_matched', seed)])))
    dense = contrast_rows(evaluator, pairs, 11, adjust='holm', label='common-preset dense-filter control (post hoc)')
    outputs['dense_common'] = dense
    outputs['dense_common_crossed'] = [c for c in (crossed(dense, a + ' minus gfnet_matched')
                                       for a in ['psan', 'psan_ad0', 'psan_gauss_ad0', 'psan_gauss_ad02']) if c]

    # Factorial (seeds 42-44), Bonferroni 24 as declared.
    cells = {'g0': 'psan_ad0', 'g2': 'psan', 'q0': 'psan_gauss_ad0', 'q2': 'psan_gauss_ad02'}
    factorial_rows = []
    complete = [s for s in range(42, 45) if all((a, s) in runs for a in cells.values())]
    effects = {}
    for seed in complete:
        scores = {c: evaluator.score(path(runs[(a, seed)]), 11) for c, a in cells.items()}
        obs = {c: v[0] for c, v in scores.items()}; dist = {c: v[1] for c, v in scores.items()}
        definitions = {
            'atom_at_ad0 (Gaussian-Gabor)': ('q0', 'g0'), 'atom_at_ad02 (Gaussian-Gabor)': ('q2', 'g2'),
            'dropout_gabor (AD0.2-AD0)': ('g2', 'g0'), 'dropout_gaussian (AD0.2-AD0)': ('q2', 'q0')}
        for name, (a, b) in definitions.items():
            factorial_rows.append(dict(effect=name, seed=seed, estimate=obs[a] - obs[b], _dist=dist[a] - dist[b]))
        joint = {
            'atom main effect (Gaussian-Gabor)': ((obs['q0'] + obs['q2'] - obs['g0'] - obs['g2']) / 2,
                                                  (dist['q0'] + dist['q2'] - dist['g0'] - dist['g2']) / 2),
            'dropout main effect (AD0.2-AD0)': ((obs['g2'] + obs['q2'] - obs['g0'] - obs['q0']) / 2,
                                                (dist['g2'] + dist['q2'] - dist['g0'] - dist['q0']) / 2),
            'interaction (difference of differences)': ((obs['q2'] - obs['q0']) - (obs['g2'] - obs['g0']),
                                                        (dist['q2'] - dist['q0']) - (dist['g2'] - dist['g0']))}
        for name, (o, d) in joint.items():
            factorial_rows.append(dict(effect=name, seed=seed, estimate=o, _dist=d))
            effects.setdefault(name, []).append((o, d))
    rng = np.random.default_rng(SEED + 11)
    picks = rng.integers(0, max(len(complete), 1), (B, max(len(complete), 1)))
    for name, values in effects.items():
        if len(values) < 2:
            continue
        stacked = np.stack([d for _, d in values])
        crossed_values = stacked[picks, np.arange(B)[:, None]].mean(1)
        estimates = np.array([o for o, _ in values])
        factorial_rows.append(dict(effect=name, seed='across ' + ','.join(map(str, complete)), estimate=float(estimates.mean()),
                                   seed_sd=float(estimates.std(ddof=1)), _dist=crossed_values))
    for r in factorial_rows:
        d = r.pop('_dist')
        r.update(ci_low=float(np.quantile(d, .025)), ci_high=float(np.quantile(d, .975)), p_uncorrected=tail(d))
        r.update(p_bonferroni=min(1., r['p_uncorrected'] * 24), family_size=24)
    outputs['factorial'] = factorial_rows

    # OFAT against the no-dropout M16 Gabor base, seed 42 (Bonferroni 10 as declared).
    variants = {'M4': ('psan_ad0_M4', 42), 'M8': ('psan_ad0_M8', 42), 'M32': ('psan_ad0_M32', 42),
                'M64': ('psan_ad0_M64', 42), 'Gaussian': ('psan_gauss_ad0', 42),
                'AD0.1': ('canine_psan_gabor_ad0p1_M16_morlet_s42', 42), 'AD0.2': ('psan', 42),
                'Axis-aligned': ('canine_psan_gabor_ad0p0_M16_morlet_axis_s42', 42),
                'Fixed phase': ('canine_psan_gabor_ad0p0_M16_morlet_fixedphase_s42', 42),
                'Random init': ('canine_psan_gabor_ad0p0_M16_random_s42', 42)}
    pairs = [(name, 42, path(runs[key]), path(runs[('psan_ad0', 42)])) for name, key in variants.items() if key in runs]
    outputs['ofat'] = contrast_rows(evaluator, pairs, 11, family=10, label='round2 OFAT family recomputed')

    # Round-3 OFAT replication: M32/M64 vs M16 (AD0, common preset), seeds 42-44, Holm 2.
    pairs = []
    for atoms in [32, 64]:
        for seed in range(42, 45):
            if (f'psan_ad0_M{atoms}', seed) in runs and ('psan_ad0', seed) in runs:
                pairs.append((f'M{atoms} minus M16', seed, path(runs[(f'psan_ad0_M{atoms}', seed)]), path(runs[('psan_ad0', seed)])))
    replication = contrast_rows(evaluator, pairs, 11, adjust='holm', label='atom-count replication')
    outputs['ofat_replication'] = replication
    replication_crossed = [c for c in (crossed(replication, f'M{m} minus M16') for m in [32, 64]) if c]
    if replication_crossed:
        adjusted = holm([c['p_crossed'] for c in replication_crossed])
        for c, value in zip(replication_crossed, adjusted):
            c.update(p_crossed_holm=float(value), family_size=len(replication_crossed))
    outputs['ofat_replication_crossed'] = replication_crossed

    # Validation-selected comparisons: round-2 seed-42 family (7, Bonferroni) and round-3 seeds 42-44 (Holm 4).
    pairs = [('Tuned PSAN minus ' + m, 42, path(tuned[('psan', 42)]), path(tuned[(m, 42)]))
             for m in ['afno', 'convnext', 'deit', 'fastvit_t8', 'gfnet', 'shvit_s1', 'swin'] if (m, 42) in tuned]
    outputs['tuned_round2'] = contrast_rows(evaluator, pairs, 11, family=7, label='round2 tuned family recomputed')
    pairs = []
    for a, b in [('psan', 'gfnet_matched'), ('psan_ad0', 'gfnet_matched'), ('psan', 'gfnet'), ('psan', 'afno')]:
        for seed in range(42, 45):
            if (a, seed) in tuned and (b, seed) in tuned:
                pairs.append((f'tuned {a} minus {b}', seed, path(tuned[(a, seed)]), path(tuned[(b, seed)])))
    tuned_rows = contrast_rows(evaluator, pairs, 11, adjust='holm', label='round3 validation-selected, per seed')
    outputs['tuned_round3'] = tuned_rows
    crossed_rows = [c for c in (crossed(tuned_rows, f'tuned {a} minus {b}') for a, b in
                    [('psan', 'gfnet_matched'), ('psan_ad0', 'gfnet_matched'), ('psan', 'gfnet'), ('psan', 'afno')]) if c]
    if crossed_rows:
        adjusted = holm([c['p_crossed'] for c in crossed_rows])
        for c, value in zip(crossed_rows, adjusted):
            c.update(p_crossed_holm=float(value), family_size=len(crossed_rows))
    outputs['tuned_round3_crossed'] = crossed_rows
    # Descriptive (outside the declared tuned family): atom dropout under validation-selected training.
    pairs = [('tuned psan minus psan_ad0', s, path(tuned[('psan', s)]), path(tuned[('psan_ad0', s)]))
             for s in range(42, 45) if ('psan', s) in tuned and ('psan_ad0', s) in tuned]
    dropout_rows = contrast_rows(evaluator, pairs, 11, adjust='holm', label='descriptive: tuned atom-dropout effect')
    outputs['tuned_dropout'] = dropout_rows
    outputs['tuned_dropout_crossed'] = [c for c in [crossed(dropout_rows, 'tuned psan minus psan_ad0')] if c]

    # Disease-only analysis: images whose reference label is not Normal; 11-way argmax, F1 over 10 diseases.
    mask = reference.label.to_numpy(int) != NORMAL
    disease = Evaluator(reference, mask=mask, seed=SEED + 3)
    pairs = []
    for b, seeds in [('gfnet_matched', range(42, 45)), ('gfnet', range(42, 47)), ('afno', range(42, 47))]:
        for a in ['psan', 'psan_ad0']:
            for seed in seeds:
                if (a, seed) in runs and (b, seed) in runs:
                    pairs.append((f'{a} minus {b}', seed, path(runs[(a, seed)]), path(runs[(b, seed)])))
    disease_rows = contrast_rows(disease, pairs, 11, labels=DISEASES, adjust='holm', label='disease-only (true label not Normal)')
    outputs['disease_contrasts'] = disease_rows
    outputs['disease_crossed'] = [c for c in (crossed(disease_rows, f'{a} minus {b}') for a in ['psan', 'psan_ad0']
                                              for b in ['gfnet_matched', 'gfnet', 'afno']) if c]
    per_model = []
    for (arm, seed), s in sorted(runs.items(), key=lambda x: (str(x[0][0]), x[0][1])):
        if arm not in ['psan', 'psan_ad0', 'psan_gauss_ad0', 'gfnet_matched', 'gfnet', 'afno'] + PRETRAINED:
            continue
        frame, logits, _ = read_predictions(path(s))
        y = frame.label.to_numpy(int)[mask]; pred = logits.argmax(1)[mask]
        conditional = logits[mask].copy(); conditional[:, NORMAL] = -np.inf
        per_model.append(dict(arm=arm, seed=seed, disease_macro_f1=observed_f1(y, pred, 11, DISEASES),
                              disease_macro_f1_conditional=observed_f1(y, conditional.argmax(1), 11, DISEASES),
                              all_class_macro_f1=s['test']['macro_f1'],
                              predicted_normal_rate=float((pred == NORMAL).mean())))
    outputs['disease_per_run'] = per_model
    return outputs


def external_contrasts(ext):
    outputs = {'rows': [], 'crossed': [], 'per_run': []}
    for dataset in sorted({d for d, _, _ in ext}):
        reference = next(s for (d, _, _), s in sorted(ext.items()) if d == dataset)
        frame, _, _ = read_predictions(summary_predictions(reference))
        evaluator = Evaluator(frame, seed=SEED + 5)
        k = len(read_predictions(summary_predictions(reference))[2])
        for (d, arm, seed), s in sorted(ext.items()):
            if d == dataset:
                observed, _ = evaluator.score(summary_predictions(s), k)
                outputs['per_run'].append(dict(dataset=d, arm=arm, seed=seed, candidate=s['candidate'],
                                               macro_f1=observed, macro_auc_ovr=s['test']['macro_auc_ovr'],
                                               accuracy=s['test']['accuracy'],
                                               best_validation_macro_f1=s['best_validation_macro_f1']))
        pairs = []
        for a, b in [('psan', 'gfnet_matched'), ('psan_ad0', 'gfnet_matched'), ('psan', 'gfnet'), ('psan', 'afno')]:
            for seed in [42, 43, 44]:
                if (dataset, a, seed) in ext and (dataset, b, seed) in ext:
                    pairs.append((f'{a} minus {b}', seed, summary_predictions(ext[(dataset, a, seed)]),
                                  summary_predictions(ext[(dataset, b, seed)])))
        rows = contrast_rows(evaluator, pairs, k, adjust='holm', label=dataset)
        for r in rows:
            r['dataset'] = dataset
        outputs['rows'] += rows
        crossed_rows = [c for c in (crossed(rows, f'{a} minus {b}') for a, b in
                        [('psan', 'gfnet_matched'), ('psan_ad0', 'gfnet_matched'), ('psan', 'gfnet'), ('psan', 'afno')]) if c]
        if crossed_rows:
            adjusted = holm([c['p_crossed'] for c in crossed_rows])
            for c, value in zip(crossed_rows, adjusted):
                c.update(dataset=dataset, p_crossed_holm=float(value), family_size=len(crossed_rows))
        outputs['crossed'] += crossed_rows
    return outputs


def arm_summary(runs, tuned, ext):
    rows = []
    for label, collection in [('common', runs), ('tuned', tuned)]:
        arms = sorted({a for a, _ in collection})
        for arm in arms:
            seeds = sorted(s for a, s in collection if a == arm)
            values = [collection[(arm, s)]['test']['macro_f1'] for s in seeds]
            vals = [collection[(arm, s)]['best_validation_macro_f1'] if 'best_validation_macro_f1' in collection[(arm, s)]
                    else collection[(arm, s)].get('val', {}).get('macro_f1') for s in seeds]
            aucs = [collection[(arm, s)]['test']['macro_auc_ovr'] for s in seeds]
            rows.append(dict(protocol=label, dataset='canine', arm=arm, seeds=','.join(map(str, seeds)), runs=len(seeds),
                             candidate=collection[(arm, seeds[0])].get('candidate', 'c0' if label == 'common' else None),
                             macro_f1_mean=float(np.mean(values)), macro_f1_sd=float(np.std(values, ddof=1)) if len(values) > 1 else None,
                             validation_macro_f1_mean=float(np.mean([v for v in vals if v is not None])) if any(v is not None for v in vals) else None,
                             macro_auc_mean=float(np.mean(aucs)),
                             parameters=collection[(arm, seeds[0])].get('parameters')))
    for dataset in sorted({d for d, _, _ in ext}):
        for arm in sorted({a for d, a, _ in ext if d == dataset}):
            seeds = sorted(s for d, a, s in ext if d == dataset and a == arm)
            values = [ext[(dataset, arm, s)]['test']['macro_f1'] for s in seeds]
            aucs = [ext[(dataset, arm, s)]['test']['macro_auc_ovr'] for s in seeds]
            rows.append(dict(protocol='tuned', dataset=dataset, arm=arm, seeds=','.join(map(str, seeds)), runs=len(seeds),
                             candidate=ext[(dataset, arm, seeds[0])]['candidate'],
                             macro_f1_mean=float(np.mean(values)), macro_f1_sd=float(np.std(values, ddof=1)) if len(values) > 1 else None,
                             validation_macro_f1_mean=float(np.mean([ext[(dataset, arm, s)]['best_validation_macro_f1'] for s in seeds])),
                             macro_auc_mean=float(np.mean(aucs)), parameters=ext[(dataset, arm, seeds[0])].get('parameters')))
    return rows


def run():
    TABLES.mkdir(parents=True, exist_ok=True)
    audit = audit_tables()
    runs, tuned, ext = canine_runs(), tuned_runs(), external_runs()
    outputs = canine_contrasts(runs, tuned)
    for name, rows in outputs.items():
        pd.DataFrame(strip(rows)).to_csv(TABLES / f'{name}.csv', index=False)
    if ext:
        external = external_contrasts(ext)
        pd.DataFrame(strip(external['rows'])).to_csv(TABLES / 'external128_contrasts.csv', index=False)
        pd.DataFrame(external['crossed']).to_csv(TABLES / 'external128_crossed.csv', index=False)
        pd.DataFrame(external['per_run']).to_csv(TABLES / 'external128_per_run.csv', index=False)
    pd.DataFrame(arm_summary(runs, tuned, ext)).to_csv(TABLES / 'arm_summary.csv', index=False)
    write_json(OUT / 'analysis_provenance.json', dict(
        bootstrap=B, bootstrap_seed=SEED, minimum_two_sided_p=2 / (B + 1),
        resampling='paired multinomial resampling of evaluation groups shared by all compared models',
        canine_groups='visual groups (not dog/visit/hospital identities)',
        across_seed='crossed bootstrap of groups and seed indices; seed-level paired t is descriptive',
        families=dict(round2_primary='Bonferroni 18', round2_ofat='Bonferroni 10', round2_factorial='Bonferroni 24',
                      round2_tuned='Bonferroni 7', round3='Holm within family'),
        post_hoc=['common-preset dense-filter controls were computed after round-2 test results were observed',
                  'disease-only analysis was added after round-2 test results were observed'],
        audit=audit, protocol_sha256=sha256(ROOT / 'configs/round3_protocol.json'),
        source_code_sha256=sha256(Path(__file__))))
    print('Round-3 analysis tables written to', TABLES, flush=True)


if __name__ == '__main__':
    run()
