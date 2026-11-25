"""Statistics for the frozen round-9 controls (dense-filter weight decay; step-matched external runs).

Every screening trial, final run and declared contrast is written to results/round9/tables. Intervals are
unadjusted 95% percentile intervals of a crossed bootstrap over test groups and the three seed indices;
p values are Holm-adjusted within the families declared in configs/round9_protocol.json.

    $PY -m experiments.round9_analysis
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .common import CLASSES, ROOT, sha256, write_json
from .round3_analysis import Evaluator, holm, read_predictions, tail
from .round9 import OUT, PROTOCOL, SEEDS, STEP_ARMS, STEP_DATASETS, WD_ARM, jobs, selection_path

TABLES = OUT / 'tables'
SUPPORT = ROOT / 'results/round7/support/final'
NAMES = {'psan': 'PSAN (AD 0.2)', 'psan_ad0': 'PSAN (AD 0)', 'gfnet_matched': 'Matched dense',
         WD_ARM: 'Matched dense, filter weight decay', 'lowrank16': 'Free rank-16'}


def crossed(estimates, distributions, picks):
    values = np.stack(distributions)[picks, np.arange(len(distributions[0]))[:, None]].mean(1)
    return dict(mean_delta=float(np.mean(estimates)), seed_sd=float(np.std(estimates, ddof=1)),
                ci_low=float(np.quantile(values, .025)), ci_high=float(np.quantile(values, .975)),
                p_uncorrected=tail(values), seeds_positive=int(np.count_nonzero(np.asarray(estimates) > 0)),
                seeds=','.join(map(str, SEEDS)), bootstrap=len(values))


def checked_summary(directory, protocol_sha, sources):
    path = directory / 'summary.json'
    item = json.loads(path.read_text())
    if item['protocol_sha256'] != protocol_sha:
        raise ValueError('Run does not belong to the frozen round-9 protocol: ' + str(path))
    if sha256(ROOT / item['checkpoint']) != item['checkpoint_sha256']:
        raise ValueError('Checkpoint changed: ' + str(path))
    sources.append(path)
    return item


def per_seed(label_a, label_b, seed, a, b, **extra):
    (oa, da), (ob, db) = a, b
    difference = da - db
    return dict(comparison=f'{label_a} minus {label_b}', seed=seed, f1_a=oa, f1_b=ob, delta=oa - ob,
                ci_low=float(np.quantile(difference, .025)), ci_high=float(np.quantile(difference, .975)),
                p_uncorrected=tail(difference), **extra)


def weight_decay(spec, sources):
    protocol_sha = sha256(PROTOCOL)
    selection = json.loads(selection_path().read_text())
    sources.append(selection_path())
    pd.DataFrame(selection['trials']).assign(arm=WD_ARM, selected=lambda f: f.candidate == selection['selected']) \
        .to_csv(TABLES / 'wd_trials.csv', index=False)
    comparators = spec['wd']['comparators']
    paths = {WD_ARM: {}, 'gfnet_matched': {}, 'psan': {}, 'psan_ad0': {}}
    run_rows = []
    for seed in SEEDS:
        directory = OUT / 'wd/final' / f'canine_{WD_ARM}_{selection["selected"]}_s{seed}_final'
        item = checked_summary(directory, protocol_sha, sources)
        paths[WD_ARM][seed] = directory / 'test_predictions.csv'
        run_rows.append(dict(arm=WD_ARM, seed=seed, candidate=item['candidate'], macro_f1=item['test']['macro_f1'],
                             validation_macro_f1=item['best_validation_macro_f1'], best_epoch=item['best_validation_epoch'],
                             epochs=item['epochs_completed']))
        paths['gfnet_matched'][seed] = ROOT / comparators['dense'].format(seed=seed) / 'test_predictions.csv'
        paths['psan'][seed] = ROOT / comparators['psan'][SEEDS.index(seed)] / 'test_predictions.csv'
        paths['psan_ad0'][seed] = ROOT / comparators['psan_ad0'].format(seed=seed) / 'test_predictions.csv'
    for arm in ['gfnet_matched', 'psan', 'psan_ad0']:
        for seed in SEEDS:
            item = json.loads(paths[arm][seed].with_name('summary.json').read_text())
            run_rows.append(dict(arm=arm, seed=seed, candidate=item['candidate'], macro_f1=item['test']['macro_f1'],
                                 validation_macro_f1=item['best_validation_macro_f1'],
                                 best_epoch=item['best_validation_epoch'], epochs=item['epochs_completed']))
    sources.extend(p for arm in paths for p in paths[arm].values())
    reference, _, classes = read_predictions(paths['gfnet_matched'][42])
    if classes != CLASSES:
        raise ValueError('Unexpected class order')
    b = spec['statistics']['bootstrap']
    evaluator = Evaluator(reference, b=b, seed=spec['statistics']['seed'])
    picks = np.random.default_rng(spec['statistics']['seed'] + 100).integers(0, len(SEEDS), (b, len(SEEDS)))
    scores = {(arm, seed): evaluator.score(paths[arm][seed], len(CLASSES)) for arm in paths for seed in SEEDS}
    for row in run_rows:
        if abs(scores[(row['arm'], row['seed'])][0] - row['macro_f1']) > 1e-10:
            raise ValueError('Saved test metric does not reproduce: ' + repr(row))
    pd.DataFrame(run_rows).to_csv(TABLES / 'wd_runs.csv', index=False)
    seed_rows, contrasts = [], []
    for a, c in [(WD_ARM, 'gfnet_matched'), ('psan', WD_ARM), ('psan_ad0', WD_ARM)]:
        rows = [per_seed(a, c, s, scores[(a, s)], scores[(c, s)]) for s in SEEDS]
        seed_rows += rows
        contrasts.append(dict(comparison=f'{a} minus {c}', **crossed(
            [r['delta'] for r in rows], [scores[(a, s)][1] - scores[(c, s)][1] for s in SEEDS], picks)))
    for row, adjusted in zip(contrasts, holm([r['p_uncorrected'] for r in contrasts])):
        row.update(p_holm=float(adjusted), family_size=3)
    pd.DataFrame(seed_rows).to_csv(TABLES / 'wd_per_seed.csv', index=False)
    pd.DataFrame(contrasts).to_csv(TABLES / 'wd_contrasts.csv', index=False)
    means = pd.DataFrame(run_rows).groupby('arm').macro_f1.agg(['mean', 'std']).reset_index()
    means.to_csv(TABLES / 'wd_aggregate.csv', index=False)
    # Not part of the family: the earlier PSAN-dense difference recomputed with this bootstrap, as a consistency record.
    reference_check = crossed([scores[('psan', s)][0] - scores[('gfnet_matched', s)][0] for s in SEEDS],
                              [scores[('psan', s)][1] - scores[('gfnet_matched', s)][1] for s in SEEDS], picks)
    print(pd.DataFrame(contrasts).to_string(index=False), flush=True)
    return dict(selected=selection['selected'], earlier_psan_minus_dense_recomputed=reference_check)


def step_matched(spec, sources):
    protocol_sha = sha256(PROTOCOL)
    run_rows, seed_rows, contrast_rows = [], [], []
    for index, dataset in enumerate(STEP_DATASETS):
        schedule = spec['steps']['schedules'][dataset]
        long, short = schedule['epochs'], 30
        items, paths = {}, {}
        for arm in STEP_ARMS:
            for seed in SEEDS:
                directory = OUT / 'steps/final' / f'{dataset}_{arm}_e{long}_c2_s{seed}_final'
                item = checked_summary(directory, protocol_sha, sources)
                if item['epochs_completed'] != long:
                    raise ValueError('Incomplete step-matched run: ' + item['tag'])
                items[(arm, seed, long)], paths[(arm, seed, long)] = item, directory / 'test_predictions.csv'
                directory = SUPPORT / f'{dataset}_{arm}_f100_c2_s{seed}_final'
                item = json.loads((directory / 'summary.json').read_text())
                if item['epochs_completed'] != short or item['support_percentage'] != 100:
                    raise ValueError('Unexpected short-schedule counterpart: ' + item['tag'])
                items[(arm, seed, short)], paths[(arm, seed, short)] = item, directory / 'test_predictions.csv'
        sources.extend(paths.values())
        reference, _, classes = read_predictions(paths[(STEP_ARMS[0], 42, long)])
        b = spec['statistics']['bootstrap']
        evaluator = Evaluator(reference, b=b, seed=spec['statistics']['seed'] + 1 + index)
        picks = np.random.default_rng(spec['statistics']['seed'] + 101 + index).integers(0, len(SEEDS), (b, len(SEEDS)))
        scores = {}
        for key, path in paths.items():
            frame, _, current = read_predictions(path)
            if current != classes or not np.array_equal(frame.group_id, reference.group_id):
                raise ValueError('Unpaired test labels/groups/classes: ' + str(path))
            scores[key] = evaluator.score(path, len(classes))
            item = items[key]
            if abs(scores[key][0] - item['test']['macro_f1']) > 1e-10:
                raise ValueError('Saved test metric does not reproduce: ' + str(path))
            arm, seed, epochs = key
            run_rows.append(dict(dataset=dataset, arm=arm, seed=seed, epochs=epochs,
                                 schedule='step-matched' if epochs == long else '30 epochs',
                                 macro_f1=scores[key][0], validation_macro_f1=item['best_validation_macro_f1'],
                                 best_epoch=item['best_validation_epoch']))
        simple, interactions, gains = [], [], []
        for control in ['gfnet_matched', 'lowrank16']:
            differences = {}
            for epochs in [long, short]:
                rows = [per_seed('psan_ad0', control, s, scores[('psan_ad0', s, epochs)], scores[(control, s, epochs)],
                                 dataset=dataset, epochs=epochs) for s in SEEDS]
                seed_rows += rows
                differences[epochs] = ([r['delta'] for r in rows],
                                       [scores[('psan_ad0', s, epochs)][1] - scores[(control, s, epochs)][1] for s in SEEDS])
            simple.append(dict(dataset=dataset, type='simple', comparison=f'psan_ad0 minus {control}', epochs=long,
                               **crossed(*differences[long], picks)))
            # The 30-epoch contrast is the published support-study cell, recomputed here with a shared bootstrap.
            contrast_rows.append(dict(dataset=dataset, type='reference_30_epochs', comparison=f'psan_ad0 minus {control}',
                                      epochs=short, **crossed(*differences[short], picks)))
            interactions.append(dict(
                dataset=dataset, type='interaction', comparison=f'(psan_ad0 minus {control}) step-matched minus 30 epochs',
                epochs=long, **crossed([x - y for x, y in zip(differences[long][0], differences[short][0])],
                                       [x - y for x, y in zip(differences[long][1], differences[short][1])], picks)))
        for family in [simple, interactions]:
            for row, adjusted in zip(family, holm([r['p_uncorrected'] for r in family])):
                row.update(p_holm=float(adjusted), family_size=2)
        for arm in STEP_ARMS:   # descriptive: change of each model with the longer schedule (no adjustment, no claim)
            gains.append(dict(dataset=dataset, type='descriptive_schedule_gain', comparison=f'{arm} step-matched minus 30 epochs',
                              epochs=long, **crossed([scores[(arm, s, long)][0] - scores[(arm, s, short)][0] for s in SEEDS],
                                                     [scores[(arm, s, long)][1] - scores[(arm, s, short)][1] for s in SEEDS], picks)))
        contrast_rows += simple + interactions + gains
        print(dataset, 'step-matched cells analyzed', flush=True)
    pd.DataFrame(run_rows).to_csv(TABLES / 'steps_runs.csv', index=False)
    pd.DataFrame(seed_rows).to_csv(TABLES / 'steps_per_seed.csv', index=False)
    pd.DataFrame(contrast_rows).to_csv(TABLES / 'steps_contrasts.csv', index=False)
    aggregate = pd.DataFrame(run_rows).groupby(['dataset', 'schedule', 'epochs', 'arm']).agg(
        mean=('macro_f1', 'mean'), std=('macro_f1', 'std'), best_epoch_min=('best_epoch', 'min'),
        best_epoch_max=('best_epoch', 'max')).reset_index()
    aggregate.to_csv(TABLES / 'steps_aggregate.csv', index=False)
    print(pd.DataFrame(contrast_rows).drop(columns=['seeds', 'bootstrap']).to_string(index=False), flush=True)


def run():
    spec = json.loads(PROTOCOL.read_text())
    TABLES.mkdir(parents=True, exist_ok=True)
    sources = []
    wd = weight_decay(spec, sources)
    step_matched(spec, sources)
    expected = len(jobs('wd-final')) + len(jobs('steps'))
    write_json(OUT / 'analysis_provenance.json', dict(
        status='COMPLETED', protocol_sha256=sha256(PROTOCOL),
        analysis_source_sha256=sha256(ROOT / 'experiments/round9_analysis.py'),
        frozen_analysis_source_sha256=spec['source_code_sha256']['experiments/round9_analysis.py'],
        completed_final_runs=expected, weight_decay=wd,
        tables={p.name: sha256(p) for p in sorted(TABLES.glob('*.csv'))},
        source_sha256={str(p.relative_to(ROOT)): sha256(p) for p in sorted(set(sources))},
        intervals='Unadjusted 95% percentile intervals; p values Holm-adjusted within the declared families.',
        stage='Disclosed post hoc controls specified before these runs; not an independent blinded confirmation.'))


if __name__ == '__main__':
    with threadpool_limits(limits=4):
        run()
