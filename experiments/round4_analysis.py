"""Round-4 statistics from saved predictions (no model training).

Declared families (configs/round4_protocol.json), Holm-adjusted within each family:
  lowrank_controls  validation-selected seeds 42-44, crossed group x seed bootstrap;
  outer_cv          pooled out-of-fold Macro F1 over all images (paired group bootstrap), with per-fold rows;
  long_training     100-epoch seed-42 runs, paired group bootstrap (test-sample variation only).
Post hoc analyses requested by the external review, labeled as such in the outputs:
  device-restricted Macro F1 (ophthalmoscope-only and camera-only test images) and
  one-versus-rest class-wise calibration before and after validation temperature scaling.
B=10,000 throughout, with the smoothed two-sided tail probability of round 3.

    $PY -m experiments.round4_analysis
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax

from evaluation_metrics import binary_expected_calibration_error
from .common import CLASSES, ROOT, sha256, write_json
from .round3_analysis import (B, SEED, Evaluator, GroupBootstrap, contrast_rows, crossed, holm, read_predictions,
                              strip, summary_predictions, tail, tuned_runs)

OUT = ROOT / 'results/round4'
TABLES = OUT / 'tables'
K = len(CLASSES)
DEVICES = {'검안경': 'Ophthalmoscope', '일반카메라': 'Conventional camera', '스마트폰': 'Smartphone'}
LOWRANK = [('psan', 'lowrank16'), ('psan_ad0', 'lowrank16'), ('psan_ad0', 'lowrank10'),
           ('psan_ad0', 'fixedatom16'), ('lowrank16', 'gfnet_matched')]
CV = [('psan', 'gfnet_matched'), ('psan_ad0', 'gfnet_matched'), ('psan', 'lowrank16'), ('psan_ad0', 'lowrank16')]
LONG = [('psan', 'gfnet_matched'), ('psan_ad0', 'gfnet_matched')]


def summaries(stage):
    runs = {}
    for path in (OUT / stage).glob('*/summary.json'):
        s = json.loads(path.read_text())
        key = (s['arm'], s['fold']) if stage == 'cv' else (s['arm'], s['seed'])
        runs[key] = s
    return runs


def selected_runs():
    """Validation-selected canine finals of all rounds keyed by (arm, seed)."""
    runs = tuned_runs()
    runs.update(summaries('final'))
    return runs


def with_crossed_holm(rows, pairs, label):
    out = [c for c in (crossed(rows, f'{a} minus {b}') for a, b in pairs) if c]
    if out:
        for c, value in zip(out, holm([c['p_crossed'] for c in out])):
            c.update(analysis=label, p_crossed_holm=float(value), family_size=len(out))
    return out


# ----------------------------------------------------------------------------- declared families

def lowrank_family(runs, evaluator):
    pairs = [(f'{a} minus {b}', s, summary_predictions(runs[(a, s)]), summary_predictions(runs[(b, s)]))
             for a, b in LOWRANK for s in (42, 43, 44) if (a, s) in runs and (b, s) in runs]
    rows = contrast_rows(evaluator, pairs, K, adjust='holm', label='lowrank controls, per seed')
    return rows, with_crossed_holm(rows, LOWRANK, 'lowrank controls, crossed')


def cv_predictions(runs, cv):
    """Out-of-fold test predictions per arm: fold 1 is the main-split seed-42 final."""
    paths = {}
    for arm in {a for pair in CV for a in pair}:
        folds = {1: summary_predictions(runs[(arm, 42)])} if (arm, 42) in runs else {}
        folds.update({f: summary_predictions(cv[(arm, f)]) for f in range(2, 6) if (arm, f) in cv})
        paths[arm] = folds
    return paths


def outer_cv(runs):
    cv = summaries('cv')
    paths = cv_predictions(runs, cv)
    fold_rows, pooled_rows = [], []
    for fold in range(1, 6):
        arms = [a for a in paths if fold in paths[a]]
        if not arms:
            continue
        reference, _, _ = read_predictions(paths[arms[0]][fold])
        evaluator = Evaluator(reference, seed=SEED + 40 + fold)
        pairs = [(f'{a} minus {b}', fold, paths[a][fold], paths[b][fold]) for a, b in CV
                 if fold in paths.get(a, {}) and fold in paths.get(b, {})]
        for row in contrast_rows(evaluator, pairs, K, adjust='holm', label=f'outer fold {fold}'):
            row['fold'] = fold
            fold_rows.append(row)
    per_arm = {}
    for arm, folds in paths.items():
        if sorted(folds) != [1, 2, 3, 4, 5]:
            continue
        frames, logits = zip(*[read_predictions(folds[f])[:2] for f in range(1, 6)])
        frame = pd.concat(frames, ignore_index=True)
        per_arm[arm] = (frame, np.concatenate(logits).argmax(1))
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
        for a, b in CV:
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
    arm_rows = [dict(arm=a, fold=f, candidate=s['candidate'], macro_f1=s['test']['macro_f1'],
                     best_validation_macro_f1=s['best_validation_macro_f1'], best_epoch=s['best_validation_epoch'])
                for (a, f), s in sorted(cv.items())]
    return fold_rows, pooled_rows, arm_rows, {a: sorted(f) for a, f in paths.items()}


def long_training(runs, evaluator):
    long = summaries('long')
    pairs = [(f'{a} minus {b}', 42, summary_predictions(long[(a, 42)]), summary_predictions(long[(b, 42)]))
             for a, b in LONG if (a, 42) in long and (b, 42) in long]
    rows = contrast_rows(evaluator, pairs, K, adjust='holm', label='100-epoch training, seed 42')
    arm_rows = []
    for (arm, seed), s in sorted(long.items()):
        thirty = runs.get((arm, seed))
        arm_rows.append(dict(arm=arm, seed=seed, best_epoch=s['best_validation_epoch'],
                             best_validation_macro_f1=s['best_validation_macro_f1'], test_macro_f1=s['test']['macro_f1'],
                             epochs30_test_macro_f1=thirty['test']['macro_f1'] if thirty else None,
                             epochs30_best_epoch=thirty.get('best_validation_epoch') if thirty else None))
    return rows, arm_rows


# ----------------------------------------------------------------------------- post hoc analyses

def device_restricted(runs, reference):
    rows, crossed_rows, arm_rows = [], [], []
    devices = reference.merge(pd.read_csv(ROOT / 'results/revision_split.csv', usecols=['path', 'device'],
                                          keep_default_na=False), on='path', how='left').device.map(DEVICES)
    if devices.isna().any():
        raise ValueError('Device missing for some test images')
    arms = ['psan', 'psan_ad0', 'gfnet_matched', 'lowrank16']
    pairs_all = [('psan', 'gfnet_matched'), ('psan_ad0', 'gfnet_matched')]
    for i, device in enumerate(['Ophthalmoscope', 'Conventional camera']):
        mask = (devices == device).to_numpy()
        labels = sorted(set(reference.label.to_numpy(int)[mask]))
        evaluator = Evaluator(reference, seed=SEED + 60 + i, mask=mask)
        for arm in arms:
            values = [evaluator.score(summary_predictions(runs[(arm, s)]), K, labels)[0]
                      for s in (42, 43, 44) if (arm, s) in runs]
            if len(values) == 3:
                arm_rows.append(dict(device=device, arm=arm, images=int(mask.sum()), classes=len(labels),
                                     macro_f1_mean=float(np.mean(values)), macro_f1_sd=float(np.std(values, ddof=1))))
        pairs = [(f'{a} minus {b}', s, summary_predictions(runs[(a, s)]), summary_predictions(runs[(b, s)]))
                 for a, b in pairs_all for s in (42, 43, 44) if (a, s) in runs and (b, s) in runs]
        device_rows = contrast_rows(evaluator, pairs, K, labels, adjust='holm', label=f'post hoc device-restricted: {device}')
        for r in device_rows:
            r['device'] = device
        rows += device_rows
        for c in with_crossed_holm(device_rows, pairs_all, f'post hoc device-restricted: {device}'):
            c.update(device=device, images=int(mask.sum()), classes=len(labels))
            crossed_rows.append(c)
    return rows, crossed_rows, arm_rows


def fit_temperature(val_logits, val_labels):
    objective = lambda logt: float(-log_softmax(val_logits / np.exp(logt), axis=1)[np.arange(len(val_labels)), val_labels].mean())
    fit = minimize_scalar(objective, bounds=(-4, 4), method='bounded')
    if not fit.success:
        raise RuntimeError('Temperature fit failed')
    return float(np.exp(fit.x))


def classwise_calibration(runs):
    rows = []
    for arm in ['psan', 'gfnet_matched']:
        summary = runs[(arm, 42)]
        directory = (ROOT / summary['checkpoint']).parent
        vf, vl, _ = read_predictions(directory / 'val_predictions.csv')
        tf, tl, _ = read_predictions(directory / 'test_predictions.csv')
        temperature = fit_temperature(vl, vf.label.to_numpy(int))
        y = tf.label.to_numpy(int)
        for condition, probabilities in [('raw', softmax(tl, axis=1)), ('temperature', softmax(tl / temperature, axis=1))]:
            for k, name in enumerate(CLASSES):
                rows.append(dict(arm=arm, seed=42, condition=condition, temperature=temperature if condition != 'raw' else 1.,
                                 cls=name, test_images=int((y == k).sum()),
                                 classwise_ece=binary_expected_calibration_error(y == k, probabilities[:, k], 15),
                                 mean_probability=float(probabilities[:, k].mean()), frequency=float((y == k).mean())))
    return rows


def learning_curves(runs):
    rows = []
    long = summaries('long')
    for source, table in [('30 epochs', runs), ('100 epochs', long)]:
        for (arm, seed), s in table.items():
            if arm not in ('psan', 'psan_ad0', 'gfnet_matched', 'lowrank16') or seed not in (42, 43, 44):
                continue
            epochs = pd.read_csv((ROOT / s['checkpoint']).parent / 'epochs.csv')
            for r in epochs.itertuples():
                rows.append(dict(budget=source, arm=arm, seed=seed, epoch=int(r.epoch), train_loss=float(r.train_loss),
                                 validation_macro_f1=float(r.validation_macro_f1), learning_rate=float(r.learning_rate)))
    return rows


# ----------------------------------------------------------------------------- main

def write(name, rows):
    frame = pd.DataFrame(strip(rows))
    frame.to_csv(TABLES / f'{name}.csv', index=False)
    return frame


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    runs = selected_runs()
    reference_path = summary_predictions(runs[('psan', 42)])
    reference, _, _ = read_predictions(reference_path)
    evaluator = Evaluator(reference, seed=SEED + 30)
    per_seed, family = lowrank_family(runs, evaluator)
    write('lowrank_per_seed', per_seed); write('lowrank_crossed', family)
    arm_summary = [dict(arm=a, seed=s, candidate=r.get('candidate'), macro_f1=r['test']['macro_f1'],
                        macro_auc=r['test']['macro_auc_ovr'], best_validation_macro_f1=r['best_validation_macro_f1'],
                        best_epoch=r.get('best_validation_epoch'),
                        filter_parameters=(r.get('filter_parameters') or {}).get('trainable_per_block'))
                   for (a, s), r in sorted(runs.items(), key=str) if s in (42, 43, 44) and
                   a in ('psan', 'psan_ad0', 'gfnet_matched', 'lowrank16', 'lowrank10', 'fixedatom16')]
    write('selected_arm_summary', arm_summary)
    fold_rows, pooled_rows, cv_arm_rows, coverage = outer_cv(runs)
    write('cv_per_fold', fold_rows); write('cv_pooled', pooled_rows); write('cv_runs', cv_arm_rows)
    long_rows, long_arms = long_training(runs, evaluator)
    write('long_contrasts', long_rows); write('long_runs', long_arms)
    device_rows, device_crossed, device_arms = device_restricted(runs, reference)
    write('device_restricted_per_seed', device_rows); write('device_restricted_crossed', device_crossed)
    write('device_restricted_arms', device_arms)
    write('classwise_calibration', classwise_calibration(runs))
    write('learning_curves', learning_curves(runs))
    selections = {p.stem: json.loads(p.read_text()) for p in (OUT / 'selections').glob('*.json')}
    write_json(OUT / 'analysis_provenance.json', dict(
        protocol_sha256=sha256(ROOT / 'configs/round4_protocol.json'), bootstrap=B, reference=str(reference_path),
        runs={'final': len(summaries('final')), 'cv': len(summaries('cv')), 'long': len(summaries('long'))},
        cv_coverage=coverage, selections={k: v['candidate']['id'] for k, v in selections.items()},
        post_hoc=['device_restricted_*', 'classwise_calibration', 'learning_curves'],
        source_sha256={p: sha256(ROOT / p) for p in ['experiments/round4_analysis.py', 'experiments/round3_analysis.py']}))
    print('Round-4 analysis tables written to', TABLES)


if __name__ == '__main__':
    main()
