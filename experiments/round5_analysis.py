"""Round-5 statistics from saved predictions (no model training).

Declared families (configs/round5_protocol.json), Holm-adjusted within each family, crossed group x seed bootstrap
over seeds 42-44 with B=10,000 and the smoothed two-sided tail probability of round 3:
  tuned_backbones   validation-selected PSAN (atom dropout 0.2) minus each validation-selected scratch backbone (primary);
  shared_backbones  shared-preset PSAN (atom dropout 0.2) minus each shared-preset scratch backbone (secondary).
Descriptive: per-arm summaries for both protocols, the screening trials, and the disease-only metrics of the shared-preset
runs (computed as in round 3).

    $PY -m experiments.round5_analysis
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .common import CLASSES, ROOT, sha256, write_json
from .round3_analysis import (DISEASES, NORMAL, SEED, Evaluator, canine_runs, contrast_rows, crossed, holm, observed_f1,
                              read_predictions, strip, summary_predictions, tuned_runs)
from .round5 import ARMS, OUT, PROTOCOL, SEEDS, SHARED

TABLES = OUT / 'tables'
K = len(CLASSES)


def round5_runs():
    """Round-5 finals keyed by (protocol, arm, seed); the shared preset is candidate c0."""
    selections = {a: json.loads((OUT / 'selections' / f'canine_{a}.json').read_text())['candidate']['id']
                  for a in ARMS if (OUT / 'selections' / f'canine_{a}.json').exists()}
    runs = {}
    for path in (OUT / 'final').glob('*/summary.json'):
        s = json.loads(path.read_text())
        if s['candidate'] == SHARED:
            runs[('shared', s['arm'], s['seed'])] = s
        if selections.get(s['arm']) == s['candidate']:
            runs[('tuned', s['arm'], s['seed'])] = s
    return runs, selections


def family(evaluator, label, prefix, reference, runs, protocol):
    pairs = [(f'{prefix}psan minus {a}', s, summary_predictions(reference[s]), summary_predictions(runs[(protocol, a, s)]))
             for a in ARMS for s in SEEDS if s in reference and (protocol, a, s) in runs]
    rows = contrast_rows(evaluator, pairs, K, adjust='holm', label=f'{label}, per seed')
    out = [c for c in (crossed(rows, f'{prefix}psan minus {a}') for a in ARMS) if c and c['runs'] == len(SEEDS)]
    if out:
        for c, value in zip(out, holm([c['p_crossed'] for c in out])):
            c.update(analysis=f'{label}, crossed', p_crossed_holm=float(value), family_size=len(out))
    return rows, out


def arm_rows(runs, selections):
    rows = []
    for (protocol, arm, seed), s in sorted(runs.items()):
        rows.append(dict(protocol=protocol, arm=arm, seed=seed, candidate=s['candidate'],
                         selected=selections.get(arm), macro_f1=s['test']['macro_f1'],
                         macro_auc=s['test']['macro_auc_ovr'], balanced_accuracy=s['test']['balanced_accuracy'],
                         best_validation_macro_f1=s['best_validation_macro_f1'], best_epoch=s['best_validation_epoch'],
                         epochs_completed=s['epochs_completed'], training_time_sec=s['training_time_sec'],
                         parameters=s['parameters'], pretrained=s['pretrained']))
    return rows


def trial_rows():
    rows = []
    for path in sorted((OUT / 'trials').glob('*/summary.json')):
        s = json.loads(path.read_text())
        if 'test' in s:
            raise ValueError('Screening trial carries a test result: ' + s['tag'])
        rows.append(dict(arm=s['arm'], candidate=s['candidate'], best_validation_macro_f1=s['best_validation_macro_f1'],
                         best_epoch=s['best_validation_epoch'], training_time_sec=s['training_time_sec']))
    return rows


def disease_rows(runs, reference):
    mask = reference.label.to_numpy(int) != NORMAL
    rows = []
    for (protocol, arm, seed), s in sorted(runs.items()):
        if protocol != 'shared':
            continue
        frame, logits, _ = read_predictions(summary_predictions(s))
        if not np.array_equal(frame.path, reference.path):
            raise ValueError('Unpaired predictions for ' + s['tag'])
        y = frame.label.to_numpy(int)[mask]; pred = logits.argmax(1)[mask]
        conditional = logits[mask].copy(); conditional[:, NORMAL] = -np.inf
        rows.append(dict(arm=arm, seed=seed, disease_macro_f1=observed_f1(y, pred, K, DISEASES),
                         disease_macro_f1_conditional=observed_f1(y, conditional.argmax(1), K, DISEASES),
                         all_class_macro_f1=s['test']['macro_f1'], predicted_normal_rate=float((pred == NORMAL).mean())))
    return rows


def write(name, rows):
    frame = pd.DataFrame(strip(rows))
    frame.to_csv(TABLES / f'{name}.csv', index=False)
    return frame


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    runs, selections = round5_runs()
    shared_psan = {s: r for (a, s), r in canine_runs().items() if a == 'psan' and s in SEEDS}
    tuned_psan = {s: r for (a, s), r in tuned_runs().items() if a == 'psan' and s in SEEDS}
    reference_path = summary_predictions(tuned_psan[42])
    reference, _, _ = read_predictions(reference_path)
    tuned_rows, tuned_family = family(Evaluator(reference, seed=SEED + 60), 'tuned backbones', 'tuned ', tuned_psan,
                                      runs, 'tuned')
    shared_rows, shared_family = family(Evaluator(reference, seed=SEED + 61), 'shared backbones', '', shared_psan,
                                        runs, 'shared')
    write('backbones_per_seed', tuned_rows + shared_rows)
    write('backbones_crossed', tuned_family + shared_family)
    write('backbone_runs', arm_rows(runs, selections))
    write('backbone_trials', trial_rows())
    write('backbone_disease_per_run', disease_rows(runs, reference))
    write_json(OUT / 'analysis_provenance.json', dict(
        protocol_sha256=sha256(PROTOCOL), bootstrap=10000, reference=str(reference_path.relative_to(ROOT)),
        runs={'shared': sum(k[0] == 'shared' for k in runs), 'tuned': sum(k[0] == 'tuned' for k in runs)},
        selections=selections,
        psan_reference_tags={'shared': [shared_psan[s]['tag'] for s in sorted(shared_psan)],
                             'tuned': [tuned_psan[s]['tag'] for s in sorted(tuned_psan)]},
        families={'tuned_backbones': len(tuned_family), 'shared_backbones': len(shared_family)},
        source_sha256={p: sha256(ROOT / p) for p in ['experiments/round5_analysis.py', 'experiments/round3_analysis.py',
                                                     'experiments/round5.py']}))
    print('Round-5 analysis tables written to', TABLES, '| runs', len(runs), '| selections', selections)


if __name__ == '__main__':
    main()
