"""Round-6 statistics from saved predictions (no model training).

Declared families (configs/round6_protocol.json), Holm-adjusted within each family, B=10,000, with the smoothed
two-sided tail probability of round 3:
  nested_cv      pooled out-of-fold Macro F1 over all 104,037 images (paired bootstrap over all visual groups), with
                 per-fold rows; folds 2-5 use the setting re-selected on each fold's own training and validation folds;
  disease_only   ten-class models trained without Normal images, seeds 42-44, crossed group x seed bootstrap;
  long_training  100-epoch runs, seeds 42-44 (seed 42 of PSAN and the dense control from round 4), crossed bootstrap;
  grid128        128-pixel models (8x8 token grid), seeds 42-44, crossed bootstrap.
Descriptive only: the selections, per-arm summaries, device-restricted Macro F1 of the disease-only models (crossed
intervals without adjustment) and the free low-rank minus dense contrast after 100 epochs.
Post hoc (configs/round6_posthoc_protocol.json, specified after the disease-only test results were known): the
disease-only comparison with c2 for every arm and 72 epochs, Holm-adjusted within its own family of 2. The nested-CV bootstrap
uses the seeds of the round-4 cross-validation, so folds whose runs are reused give identical per-fold results.

    $PY -m experiments.round6_analysis
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .common import CLASSES, ROOT, sha256, write_json
from .round3_analysis import (B, SEED, Evaluator, GroupBootstrap, contrast_rows, crossed, holm, read_predictions, strip,
                              summary_predictions, tail)
from .round4_analysis import DEVICES, selected_runs, summaries as round4_summaries
from .round6 import DISEASES, NCV_ARMS, NCV_FOLDS, NCV_REUSE, OUT, PROTOCOL, SEEDS

TABLES = OUT / 'tables'
POSTHOC = ROOT / 'results/round6_posthoc/disease/final'
K = len(CLASSES)
NCV = [('psan', 'gfnet_matched'), ('psan_ad0', 'gfnet_matched'), ('psan', 'lowrank16'), ('psan_ad0', 'lowrank16')]
DIS = [('psan', 'gfnet_matched'), ('psan', 'lowrank16')]
LONG = [('psan', 'gfnet_matched'), ('psan', 'lowrank16')]
G128 = [('psan', 'gfnet_matched')]


def load(path):
    return json.loads(path.read_text())


def finals(group):
    base = OUT / group if group == 'long' else OUT / group / 'final'
    return {(s['arm'], s['seed']): s for s in (load(p) for p in base.glob('*/summary.json'))}


def run_rows(runs, experiment):
    return [dict(experiment=experiment, arm=a, seed=s, candidate=r['candidate'], macro_f1=r['test']['macro_f1'],
                 balanced_accuracy=r['test']['balanced_accuracy'], macro_auc=r['test']['macro_auc_ovr'],
                 best_validation_macro_f1=r['best_validation_macro_f1'], best_epoch=r['best_validation_epoch'],
                 epochs_completed=r['epochs_completed'], parameters=r.get('parameters'), tag=r['tag'])
            for (a, s), r in sorted(runs.items())]


def seed_family(runs, pairs, k, evaluator, label, adjust=True, labels=None):
    items = [(f'{a} minus {b}', s, summary_predictions(runs[(a, s)]), summary_predictions(runs[(b, s)]))
             for a, b in pairs for s in SEEDS if (a, s) in runs and (b, s) in runs]
    rows = contrast_rows(evaluator, items, k, labels, adjust='holm', label=f'{label}, per seed')
    out = [c for c in (crossed(rows, f'{a} minus {b}') for a, b in pairs) if c and c['runs'] == len(SEEDS)]
    if out and adjust:
        for c, value in zip(out, holm([c['p_crossed'] for c in out])):
            c.update(analysis=f'{label}, crossed', p_crossed_holm=float(value), family_size=len(out))
    for c in out:
        c.setdefault('analysis', f'{label}, crossed (descriptive, not adjusted)')
    return rows, out


# ----------------------------------------------------------------------------- nested cross-validation

def nested_runs():
    cv, rows = {}, []
    for arm in NCV_ARMS:
        for fold in NCV_FOLDS:
            path = OUT / 'selections' / f'ncv_{arm}_fold{fold}.json'
            if not path.exists():
                continue
            selection = load(path)
            candidate = selection['candidate']['id']
            reused = candidate == NCV_REUSE['candidate']
            directory = (ROOT / NCV_REUSE['directory'].format(arm=arm, fold=fold) if reused
                         else OUT / 'ncv' / 'final' / f'canine_{arm}_{candidate}_s42_ncv{fold}')
            summary = directory / 'summary.json'
            row = dict(arm=arm, fold=fold, candidate=candidate, reused_round4=reused,
                       screen_c0=selection['validation_scores'][0], screen_c1=selection['validation_scores'][1],
                       screen_c2=selection['validation_scores'][2], run=str(directory.relative_to(ROOT)),
                       available=summary.exists())
            if summary.exists():
                s = load(summary)
                if s['fold'] != fold or s['candidate'] != candidate or s['seed'] != 42:
                    raise ValueError('Nested-CV run does not match its selection: ' + str(directory))
                cv[(arm, fold)] = s
                row.update(macro_f1=s['test']['macro_f1'], best_validation_macro_f1=s['best_validation_macro_f1'])
            rows.append(row)
    return cv, rows


def nested_cv(main_runs):
    """As round4_analysis.outer_cv, with folds 2-5 taken from the nested selection."""
    cv, selection_rows = nested_runs()
    paths = {}
    for arm in {a for pair in NCV for a in pair}:
        folds = {1: summary_predictions(main_runs[(arm, 42)])} if (arm, 42) in main_runs else {}
        folds.update({f: summary_predictions(cv[(arm, f)]) for f in NCV_FOLDS if (arm, f) in cv})
        paths[arm] = folds
    fold_rows, pooled_rows = [], []
    for fold in range(1, 6):
        arms = [a for a in paths if fold in paths[a]]
        if not arms:
            continue
        reference, _, _ = read_predictions(paths[arms[0]][fold])
        evaluator = Evaluator(reference, seed=SEED + 40 + fold)
        pairs = [(f'{a} minus {b}', fold, paths[a][fold], paths[b][fold]) for a, b in NCV
                 if fold in paths.get(a, {}) and fold in paths.get(b, {})]
        for row in contrast_rows(evaluator, pairs, K, adjust='holm', label=f'nested outer fold {fold}'):
            row['fold'] = fold
            fold_rows.append(row)
    per_arm = {}
    for arm, folds in paths.items():
        if sorted(folds) != [1, 2, 3, 4, 5]:
            continue
        frames, logits = zip(*[read_predictions(folds[f])[:2] for f in range(1, 6)])
        per_arm[arm] = (pd.concat(frames, ignore_index=True), np.concatenate(logits).argmax(1))
    if per_arm:
        reference = next(iter(per_arm.values()))[0]
        if reference.path.duplicated().any() or len(reference) != 104037:
            raise ValueError('Out-of-fold predictions must cover every image once')
        boot = GroupBootstrap(reference.group_id.to_numpy(), B, SEED + 50)
        y = reference.label.to_numpy(int)
        scores = {}
        for arm, (frame, pred) in per_arm.items():
            if not (np.array_equal(frame.path, reference.path) and np.array_equal(frame.label, reference.label)):
                raise ValueError('Unpaired out-of-fold predictions for ' + arm)
            scores[arm] = boot.macro_f1(y, pred, K)
        for a, b in NCV:
            if a in scores and b in scores:
                difference = scores[a][1] - scores[b][1]
                per_fold = [r['delta_macro_f1'] for r in fold_rows if r['comparison'] == f'{a} minus {b}']
                pooled_rows.append(dict(comparison=f'{a} minus {b}', f1_a=scores[a][0], f1_b=scores[b][0],
                                        delta_macro_f1=scores[a][0] - scores[b][0],
                                        ci_low=float(np.quantile(difference, .025)),
                                        ci_high=float(np.quantile(difference, .975)), p_uncorrected=tail(difference),
                                        folds=len(per_fold), folds_positive=int(np.sum(np.asarray(per_fold) > 0)),
                                        fold_mean_delta=float(np.mean(per_fold)) if per_fold else None,
                                        fold_sd_delta=float(np.std(per_fold, ddof=1)) if len(per_fold) > 1 else None))
        if pooled_rows:
            for r, value in zip(pooled_rows, holm([r['p_uncorrected'] for r in pooled_rows])):
                r.update(p_holm=float(value), family_size=len(pooled_rows))
    return selection_rows, fold_rows, pooled_rows, {a: sorted(f) for a, f in paths.items()}


# ----------------------------------------------------------------------------- disease-only training

def disease_devices(runs, reference):
    devices = reference.merge(pd.read_csv(ROOT / 'results/revision_split.csv', usecols=['path', 'device'],
                                          keep_default_na=False), on='path', how='left').device.map(DEVICES)
    if devices.isna().any() or len(devices) != len(reference):
        raise ValueError('Device missing for some test images')
    arm_rows, crossed_rows = [], []
    for i, device in enumerate(['Smartphone', 'Ophthalmoscope', 'Conventional camera']):
        mask = (devices == device).to_numpy()
        labels = sorted(set(reference.label.to_numpy(int)[mask]))
        evaluator = Evaluator(reference, seed=SEED + 80 + i, mask=mask)
        for arm in sorted({a for a, _ in runs}):
            values = [evaluator.score(summary_predictions(runs[(arm, s)]), len(DISEASES), labels)[0]
                      for s in SEEDS if (arm, s) in runs]
            if len(values) == len(SEEDS):
                arm_rows.append(dict(device=device, arm=arm, images=int(mask.sum()), classes=len(labels),
                                     macro_f1_mean=float(np.mean(values)), macro_f1_sd=float(np.std(values, ddof=1))))
        _, out = seed_family(runs, DIS, len(DISEASES), evaluator, f'disease-only, {device} test images', adjust=False,
                             labels=labels)
        for c in out:
            c.update(device=device, images=int(mask.sum()), classes=len(labels))
            crossed_rows.append(c)
    return arm_rows, crossed_rows


# ----------------------------------------------------------------------------- main

def write(name, rows):
    frame = pd.DataFrame(strip(rows))
    frame.to_csv(TABLES / f'{name}.csv', index=False)
    return frame


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    main_runs = selected_runs()
    outputs = {}

    selection_rows, fold_rows, pooled_rows, coverage = nested_cv(main_runs)
    write('ncv_selections', selection_rows); write('ncv_per_fold', fold_rows); write('ncv_pooled', pooled_rows)
    outputs['nested_cv'] = dict(folds=coverage, pooled=len(pooled_rows))

    dis = finals('disease')
    if dis:
        reference, _, _ = read_predictions(summary_predictions(dis[min(dis)]))
        rows, family = seed_family(dis, DIS, len(DISEASES), Evaluator(reference, seed=SEED + 70), 'disease-only')
        write('disease_per_seed', rows); write('disease_crossed', family); write('disease_runs', run_rows(dis, 'dis'))
        device_arms, device_crossed = disease_devices(dis, reference)
        write('disease_device_arms', device_arms); write('disease_device_crossed', device_crossed)
        outputs['disease_only'] = dict(runs=len(dis), family=len(family))

    # post hoc (configs/round6_posthoc_protocol.json): disease-only, c2 for every arm, 72 epochs (step-matched)
    dis72 = {(s['arm'], s['seed']): s for s in (load(p) for p in POSTHOC.glob('*/summary.json'))}
    if dis72:
        reference, _, _ = read_predictions(summary_predictions(dis72[min(dis72)]))
        rows, family = seed_family(dis72, DIS, len(DISEASES), Evaluator(reference, seed=SEED + 110),
                                   'post hoc disease-only, 72 epochs')
        write('disease72_per_seed', rows); write('disease72_crossed', family); write('disease72_runs', run_rows(dis72, 'dis72'))
        device_arms, device_crossed = disease_devices(dis72, reference)
        write('disease72_device_arms', device_arms); write('disease72_device_crossed', device_crossed)
        outputs['disease72_posthoc'] = dict(runs=len(dis72), family=len(family))

    long = {k: v for k, v in round4_summaries('long').items() if k[0] in ('psan', 'gfnet_matched') and k[1] == 42}
    long.update(finals('long'))
    if ('psan', 42) in long:
        reference, _, _ = read_predictions(summary_predictions(long[('psan', 42)]))
        evaluator = Evaluator(reference, seed=SEED + 90)
        rows, family = seed_family(long, LONG, K, evaluator, '100-epoch training')
        extra_rows, extra = seed_family(long, [('lowrank16', 'gfnet_matched')], K, evaluator,
                                        '100-epoch training', adjust=False)
        write('long_per_seed', rows + extra_rows); write('long_crossed', family + extra)
        arm_rows = run_rows(long, 'long')
        for r in arm_rows:
            thirty = main_runs.get((r['arm'], r['seed']))
            r.update(epochs30_macro_f1=thirty['test']['macro_f1'] if thirty else None,
                     epochs30_best_epoch=thirty.get('best_validation_epoch') if thirty else None)
        write('long_runs', arm_rows)
        outputs['long_training'] = dict(runs=len(long), family=len(family))

    g128 = finals('grid128')
    if g128:
        reference, _, _ = read_predictions(summary_predictions(g128[min(g128)]))
        rows, family = seed_family(g128, G128, K, Evaluator(reference, seed=SEED + 100), '128-pixel input')
        write('grid128_per_seed', rows); write('grid128_crossed', family); write('grid128_runs', run_rows(g128, 'g128'))
        outputs['grid128'] = dict(runs=len(g128), family=len(family))

    selections = {p.stem: load(p) for p in sorted((OUT / 'selections').glob('*.json'))}
    write('selections', [dict(unit=k, candidate=v['candidate']['id'], screen_c0=v['validation_scores'][0],
                              screen_c1=v['validation_scores'][1], screen_c2=v['validation_scores'][2])
                         for k, v in selections.items()])
    write_json(OUT / 'analysis_provenance.json', dict(
        protocol_sha256=sha256(PROTOCOL), bootstrap=B, outputs=outputs,
        selections={k: v['candidate']['id'] for k, v in selections.items()},
        source_sha256={p: sha256(ROOT / p) for p in ['experiments/round6_analysis.py', 'experiments/round4_analysis.py',
                                                     'experiments/round3_analysis.py', 'experiments/round6.py']}))
    print('Round-6 analysis tables written to', TABLES, json.dumps(outputs))


if __name__ == '__main__':
    main()
