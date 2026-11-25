"""Exact-raster sensitivity of the original validation-selected Blood benchmark.

This post hoc analysis changes no classifier, selection or original split. It
reuses the already frozen outcome-independent mask from the round7 support
sensitivity, but applies it to Table15's distinct selected-model predictions.
All five arms and three seeds are retained; no GPU or new training is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits

from .common import ROOT, sha256, write_json
from .round3_analysis import Evaluator, contrast_rows, crossed, holm, read_predictions, strip

OUT = ROOT/'results/round8/external_sensitivity'
PROTOCOL = ROOT/'configs/round8_external_sensitivity_protocol.json'
MASK = ROOT/'results/round7/support_sensitivity/bloodmnist128_test_mask.csv'
DATA = ROOT/'results/external_data/round3_128'
ARMS = ['psan','psan_ad0','gfnet_matched','gfnet','afno']
SEEDS = [42,43,44]
PAIRS = [('psan','gfnet_matched'),('psan_ad0','gfnet_matched'),('psan','gfnet'),('psan','afno')]
SOURCES = ['experiments/round8_external_sensitivity.py','experiments/round3_analysis.py','experiments/common.py']


def inventory():
    runs={}
    for arm in ARMS:
        for seed in SEEDS:
            paths=list((ROOT/'results/round3/external/final').glob(f'bloodmnist128_{arm}_c[0-9]_s{seed}_final/summary.json'))
            if len(paths)!=1: raise ValueError(f'Expected one selected final for {arm}/{seed}: {paths}')
            path=paths[0];summary=json.loads(path.read_text())
            if summary['dataset']!='bloodmnist128' or summary['arm']!=arm or summary['seed']!=seed:
                raise ValueError('Selected source identifiers differ')
            if sha256(ROOT/summary['checkpoint'])!=summary['checkpoint_sha256']:
                raise ValueError('Selected checkpoint changed')
            runs[(arm,seed)]=(path,summary)
    return runs


def audit_mask():
    """Recompute exact raster membership without reading classifier outcomes."""
    provenance=json.loads((DATA/'bloodmnist128_provenance.json').read_text())
    npz=DATA/'bloodmnist_128.npz'
    if sha256(npz)!=provenance['official_sha256']:raise ValueError('Official Blood file changed')
    mask=pd.read_csv(MASK,keep_default_na=False)
    with np.load(npz) as arrays:
        hashes={split:[hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
                       for x in arrays[split+'_images']] for split in ['train','val','test']}
        labels=arrays['test_labels'].reshape(-1).astype(np.int64)
    sets={key:set(value) for key,value in hashes.items()}
    exclude=np.array([digest in sets['train']|sets['val'] for digest in hashes['test']])
    expected=pd.DataFrame(dict(path=[f'bloodmnist128/test/{i}' for i in range(len(labels))],
        label=labels,group_id=[f'bloodmnist128/test/{i}' for i in range(len(labels))],
        pixel_sha256=hashes['test'],excluded=exclude))
    if not mask.equals(expected):raise ValueError('Frozen mask differs from independently recomputed rasters')
    overlaps={a+'_'+b:len(sets[a]&sets[b]) for a,b in [('train','val'),('train','test'),('val','test')]}
    if overlaps!=provenance['pixel_hash_cross_split_overlap']:raise ValueError('Blood overlap counts differ')
    if len(mask)!=3421 or exclude.sum()!=6:raise ValueError('Unexpected frozen exclusion count')
    return dict(original_images=len(mask),excluded_images=int(exclude.sum()),retained_images=int((~exclude).sum()),
                cross_split_shared_unique_rasters=overlaps,mask_recomputed_from_official_rasters=True,
                within_training_duplicate_copies=len(hashes['train'])-len(sets['train']))


def freeze():
    if PROTOCOL.exists():raise FileExistsError('Protocol is already frozen; use a new version instead of overwriting')
    OUT.mkdir(parents=True,exist_ok=True)
    audit=audit_mask();runs=inventory();inputs={}
    for path,summary in runs.values():
        for source in [path,path.with_name('test_predictions.csv')]:
            inputs[str(source.relative_to(ROOT))]=sha256(source)
    for source in [MASK,DATA/'bloodmnist_128.npz',DATA/'bloodmnist128_provenance.json',
                   DATA/'octmnist128_provenance.json',ROOT/'results/round3/tables/external128_crossed.csv']:
        inputs[str(source.relative_to(ROOT))]=sha256(source)
    oct_provenance=json.loads((DATA/'octmnist128_provenance.json').read_text())
    write_json(PROTOCOL,dict(protocol='round8_selected_blood_exact_raster_sensitivity_v1',
        frozen_kst=datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),post_hoc=True,not_a_blinded_holdout=True,
        prior_exposure='All submitted and round7 outcomes and the new rereview were known. This analysis was specified before computing the selected-model scores under the reused mask.',
        scope='Original Table15 validation-selected BloodMNIST128 models; distinct from Table16 fixed-recipe support runs.',
        rule='Reuse the frozen round7 mask excluding any test raster present in the FULL training or validation source; one identical mask for every selected model/seed.',
        arms=ARMS,seeds=SEEDS,contrasts=[a+' minus '+b for a,b in PAIRS],
        validation_and_training='Unchanged; no refitting, reselection, recalibration or retraining.',
        statistics=dict(bootstrap=10000,group_seed=2031,crossed_seed=2033,
                        grouping='individual official test image; no patient identifiers',
                        family='Same four selected-model contrasts as the original Blood benchmark; Holm4 separately for original and retained evaluations.',
                        per_seed='Descriptive paired percentile intervals, not per-seed significance claims',
                        primary='Report all15 run scores and all4 crossed contrasts, irrespective of direction/significance.'),
        audit=audit,oct_context=dict(provenance=str((DATA/'octmnist128_provenance.json').relative_to(ROOT)),
            overlap_semantics='Counts of intersecting sets of SHA256 hashes of decoded uint8 image rasters, not numbers of patient identities or duplicate pairs.',
            cross_split_shared_unique_rasters=oct_provenance['pixel_hash_cross_split_overlap'],
            patient_or_lesion_ids_in_npz=oct_provenance['patient_or_lesion_ids_in_npz']),
        limitations='Test-only exclusion does not repair training-validation overlap or unknown patient dependence and is not independent confirmation.',
        inputs=inputs,source_code_sha256={p:sha256(ROOT/p) for p in SOURCES}))
    write_json(OUT/'raster_audit.json',audit)
    print('Frozen',sha256(PROTOCOL),audit,flush=True)


def verify():
    spec=json.loads(PROTOCOL.read_text())
    for relative,digest in {**spec['inputs'],**spec['source_code_sha256']}.items():
        if sha256(ROOT/relative)!=digest:raise ValueError('Frozen source/input changed: '+relative)
    return spec


def run():
    spec=verify();runs=inventory();mask=pd.read_csv(MASK,keep_default_na=False);keep=~mask.excluded.to_numpy(bool)
    reference,_,classes=read_predictions(runs[('psan',42)][0].with_name('test_predictions.csv'))
    if not reference.equals(mask[['path','label','group_id']]):raise ValueError('Selected predictions and mask do not align')
    evals={name:Evaluator(reference,b=10000,seed=2031,mask=retain) for name,retain in
           [('original',np.ones(len(reference),bool)),('raster_excluded',keep)]}
    scores={};run_rows=[];per_seed=[];contrasts=[]
    for arm in ARMS:
        for seed in SEEDS:
            path,summary=runs[(arm,seed)];prediction=path.with_name('test_predictions.csv')
            frame,logits,current_classes=read_predictions(prediction)
            if current_classes!=classes or not frame.equals(reference):raise ValueError('Unpaired selected predictions')
            original=evals['original'].score(prediction,len(classes))[0]
            retained=evals['raster_excluded'].score(prediction,len(classes))[0]
            if abs(original-summary['test']['macro_f1'])>1e-12:raise ValueError('Original selected F1 does not reproduce')
            independent=float(f1_score(frame.label.to_numpy()[keep],logits.argmax(1)[keep],
                                       labels=range(len(classes)),average='macro',zero_division=0))
            if abs(retained-independent)>1e-12:raise ValueError('Independent masked sklearn score differs')
            run_rows.append(dict(arm=arm,seed=seed,candidate=summary['candidate'],original_macro_f1=original,
                retained_macro_f1=retained,delta_due_to_exclusion=retained-original,
                original_test_images=len(reference),retained_test_images=int(keep.sum()),excluded_test_images=int((~keep).sum())))
    for name,evaluator in evals.items():
        pairs=[(a+' minus '+b,seed,runs[(a,seed)][0].with_name('test_predictions.csv'),
                runs[(b,seed)][0].with_name('test_predictions.csv')) for a,b in PAIRS for seed in SEEDS]
        rows=contrast_rows(evaluator,pairs,len(classes),adjust='holm',label=name)
        values=[crossed(rows,a+' minus '+b,seed=2033) for a,b in PAIRS]
        for value,p in zip(values,holm([v['p_crossed'] for v in values])):
            value.update(evaluation=name,p_holm=float(p),family_size=4)
        contrasts.extend(values)
        for row in rows:
            row.pop('p_adjusted',None);row.pop('adjustment',None);row.pop('family_size',None)
            row['evaluation']=name
        per_seed.extend(strip(rows))
    prior=pd.read_csv(ROOT/'results/round3/tables/external128_crossed.csv')
    prior=prior[prior.dataset=='bloodmnist128'].set_index('comparison')
    for row in contrasts:
        if row['evaluation']!='original':continue
        old=prior.loc[row['comparison']]
        for new_key,old_key in [('mean_delta','mean_delta'),('crossed_ci_low','crossed_ci_low'),
                                ('crossed_ci_high','crossed_ci_high'),('p_crossed','p_crossed'),('p_holm','p_crossed_holm')]:
            if abs(row[new_key]-old[old_key])>1e-10:raise ValueError('Original crossed contrast does not reproduce: '+new_key)
    tables=OUT/'tables';tables.mkdir(exist_ok=True)
    frame=pd.DataFrame(run_rows);frame.to_csv(tables/'per_run.csv',index=False)
    aggregate=frame.groupby('arm')[['original_macro_f1','retained_macro_f1']].agg(['mean','std'])
    aggregate.columns=['_'.join(c) for c in aggregate.columns];aggregate.reset_index().to_csv(tables/'aggregate.csv',index=False)
    pd.DataFrame(per_seed).to_csv(tables/'per_seed_contrasts.csv',index=False)
    comparison=pd.DataFrame(contrasts);comparison.to_csv(tables/'crossed_contrasts.csv',index=False)
    original=comparison[comparison.evaluation=='original'].set_index('comparison')
    retained=comparison[comparison.evaluation=='raster_excluded'].set_index('comparison')
    changed=(original.p_holm.lt(.05)!=retained.p_holm.lt(.05)).any()
    proof=dict(status='COMPLETED',protocol_sha256=sha256(PROTOCOL),source_code_sha256=spec['source_code_sha256'],
        inputs=spec['inputs'],audit=spec['audit'],oct_context=spec['oct_context'],completed_selected_runs=15,
        original_points_and_crossed_statistics_reproduced=True,independent_masked_sklearn_scores_match=True,
        all_declared_contrasts_reported=True,significance_conclusions_changed=bool(changed),
        maximum_absolute_run_f1_change=float(frame.delta_due_to_exclusion.abs().max()),
        tables={str(p.relative_to(ROOT)):sha256(p) for p in tables.glob('*.csv')},
        limitations=spec['limitations'])
    write_json(OUT/'analysis_provenance.json',proof)
    print(aggregate.to_string(),flush=True)
    print(comparison.to_string(index=False),flush=True)
    print('Maximum F1 change:',proof['maximum_absolute_run_f1_change'],'conclusions changed:',changed,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['freeze','run'])
    args=parser.parse_args()
    with threadpool_limits(limits=4):
        freeze() if args.command=='freeze' else run()
