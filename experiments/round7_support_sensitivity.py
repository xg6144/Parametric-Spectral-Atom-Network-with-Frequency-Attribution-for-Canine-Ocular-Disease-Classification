"""Disclosed exact-raster-overlap sensitivity of the frozen support experiment.

The primary official/corrected splits and all54 training runs stay unchanged.
One outcome-independent test mask per domain excludes any raster also present in
the FULL training or validation set. The same mask is used for every support,
model and seed. Unknown patient dependence and training/validation overlap remain.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .common import ROOT, sha256, write_json
from .round3_analysis import Evaluator, holm, read_predictions
from .round7_support import ARMS, DATASETS, FRACTIONS, SEEDS, OUT as PRIMARY, PROTOCOL as PRIMARY_PROTOCOL
from .round7_support_analysis import summarize

OUT = ROOT / 'results/round7/support_sensitivity'
PROTOCOL = ROOT / 'configs/round7_support_sensitivity_protocol.json'
SOURCES = ['experiments/round7_support_sensitivity.py', 'experiments/round7_support.py',
           'experiments/round7_support_analysis.py', 'experiments/round3_analysis.py', 'experiments/common.py']


def raster_hashes(images):
    return [hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest() for image in images]


def freeze():
    if PROTOCOL.exists():
        raise FileExistsError('Sensitivity protocol already frozen; do not overwrite it.')
    OUT.mkdir(parents=True, exist_ok=True)
    records = {}; audit_rows = []
    for dataset in DATASETS:
        base = ROOT / 'results/external_data/round3_128'
        provenance_path = base / f'{dataset}_provenance.json'
        provenance = json.loads(provenance_path.read_text())
        filename = provenance['prepared_file'] if dataset == 'dermamnist_c128' else 'bloodmnist_128.npz'
        npz_path = base / filename
        digest = sha256(npz_path)
        expected = provenance['prepared_sha256'] if dataset == 'dermamnist_c128' else provenance['official_sha256']
        if digest != expected: raise ValueError('External rasters changed.')
        with np.load(npz_path) as arrays:
            hashes = {split:raster_hashes(arrays[f'{split}_images']) for split in ['train','val','test']}
            sets = {split:set(values) for split,values in hashes.items()}
            labels = arrays['test_labels'].reshape(-1)
            paths = [f'{dataset}/test/{i}' for i in range(len(labels))]
            groups = arrays['test_groups'] if 'test_groups' in arrays.files else paths
            source = sets['train'] | sets['val']
            excluded = np.array([h in source for h in hashes['test']], dtype=bool)
            mask = pd.DataFrame(dict(path=paths, label=labels, group_id=groups,
                                     pixel_sha256=hashes['test'], excluded=excluded))
        mask_path = OUT / f'{dataset}_test_mask.csv'; mask.to_csv(mask_path,index=False)
        row = dict(dataset=dataset, train_images=len(hashes['train']), validation_images=len(hashes['val']),
                   test_images=len(hashes['test']), retained_test_images=int((~excluded).sum()),
                   excluded_test_images=int(excluded.sum()),
                   train_unique_rasters=len(sets['train']), validation_unique_rasters=len(sets['val']),
                   test_unique_rasters=len(sets['test']),
                   train_validation_shared_unique_rasters=len(sets['train'] & sets['val']),
                   train_test_shared_unique_rasters=len(sets['train'] & sets['test']),
                   validation_test_shared_unique_rasters=len(sets['val'] & sets['test']))
        audit_rows.append(row)
        records[dataset] = dict(npz=str(npz_path.relative_to(ROOT)), npz_sha256=digest,
                               provenance=str(provenance_path.relative_to(ROOT)), provenance_sha256=sha256(provenance_path),
                               mask=str(mask_path.relative_to(ROOT)), mask_sha256=sha256(mask_path),
                               audit=row, excluded_by_class={str(k):int(v) for k,v in mask[mask.excluded].groupby('label').size().items()})
    pd.DataFrame(audit_rows).to_csv(OUT/'raster_overlap_audit.csv',index=False)
    completed = len(list((PRIMARY/'final').glob('*/summary.json')))
    write_json(PROTOCOL, dict(protocol='round7_common_exact_raster_exclusion_v1',
        frozen_kst=datetime.now(ZoneInfo('Asia/Seoul')).isoformat(), post_hoc=True, not_a_blinded_holdout=True,
        primary_protocol_sha256=sha256(PRIMARY_PROTOCOL), primary_completed_cells_at_freeze=completed,
        prior_exposure='Submitted results and official Blood raster-overlap counts were known; primary production was in progress and some outcomes may already have existed. Mask selection never uses classifier outcomes.',
        mask_rule='Exclude each test raster whose exact decoded uint8 raster SHA256 occurs in ANY full training or validation image; same mask across all models, fractions and seeds.',
        domains='Apply identical audit and exclusion rule to both BloodMNIST128 and corrected DermaMNIST128, including zero-exclusion domains.',
        training='No retraining, no new hyperparameter selection, no alteration of primary54runs or official/corrected splits.',
        statistics='B10000 crossed test-group by support/initialization seed bootstrap; same six simple and four interaction contrasts per domain as primary, separately Holm-adjusted for this sensitivity.',
        interpretation='Descriptive robustness within an existing cohort; no independent confirmation, patient independence or repair of training-validation overlap.',
        data=records, source_code_sha256={p:sha256(ROOT/p) for p in SOURCES}))
    print(pd.DataFrame(audit_rows).to_string(index=False),flush=True)
    print('Sensitivity frozen before reading classifier predictions for this analysis.',flush=True)


def verify():
    spec=json.loads(PROTOCOL.read_text())
    if sha256(PRIMARY_PROTOCOL)!=spec['primary_protocol_sha256']: raise ValueError('Primary protocol changed.')
    for relative,digest in spec['source_code_sha256'].items():
        if sha256(ROOT/relative)!=digest: raise ValueError('Sensitivity source changed: '+relative)
    for item in spec['data'].values():
        for name in ['npz','provenance','mask']:
            if sha256(ROOT/item[name])!=item[f'{name}_sha256']: raise ValueError('Sensitivity input changed: '+item[name])
    return spec


def run():
    spec=verify()
    primary_provenance=json.loads((PRIMARY/'analysis_provenance.json').read_text())
    if primary_provenance['completed_cells']!=54: raise ValueError('Primary analysis is incomplete.')
    for relative,digest in primary_provenance['source_sha256'].items():
        if sha256(ROOT/relative)!=digest: raise ValueError('Primary prediction/summary changed.')
    results={}
    for path in sorted((PRIMARY/'final').glob('*/summary.json')):
        item=json.loads(path.read_text())
        key=(item['dataset'],item['arm'],item['seed'],item['support_percentage'])
        if item['smoke'] or item['epochs_completed']!=30 or item['protocol_sha256']!=sha256(PRIMARY_PROTOCOL):
            raise ValueError('Incomplete or mismatched primary run.')
        if key in results: raise ValueError('Duplicated primary cell.')
        results[key]=item
    expected={(d,a,s,f) for d in DATASETS for a in ARMS for s in SEEDS for f in FRACTIONS}
    if set(results)!=expected: raise ValueError('All54 primary cells are required.')
    rows=[]; simple_rows=[]; interaction_rows=[]
    for domain_index,dataset in enumerate(DATASETS):
        mask_frame=pd.read_csv(ROOT/spec['data'][dataset]['mask'],keep_default_na=False)
        keep=~mask_frame.excluded.to_numpy(dtype=bool)
        item=results[(dataset,'psan_ad0',42,10)]
        reference,_,classes=read_predictions((ROOT/item['checkpoint']).with_name('test_predictions.csv'))
        if not reference[['path','label','group_id']].equals(mask_frame[['path','label','group_id']]):
            raise ValueError('Frozen mask is not aligned with primary test predictions.')
        evaluator=Evaluator(reference,b=10000,seed=20261109+domain_index,mask=keep)
        picks=np.random.default_rng(20261209+domain_index).integers(0,len(SEEDS),(10000,len(SEEDS)))
        scores={}
        for arm in ARMS:
            for fraction in FRACTIONS:
                for seed in SEEDS:
                    item=results[(dataset,arm,seed,fraction)]
                    path=(ROOT/item['checkpoint']).with_name('test_predictions.csv')
                    score,distribution=evaluator.score(path,len(classes))
                    scores[(arm,seed,fraction)]=(score,distribution)
                    rows.append(dict(dataset=dataset,arm=arm,seed=seed,fraction=fraction,
                                     primary_macro_f1=item['test']['macro_f1'],retained_macro_f1=score,
                                     delta_due_to_exclusion=score-item['test']['macro_f1'],
                                     retained_test_images=int(keep.sum()),excluded_test_images=int((~keep).sum())))
        simple=[];differences={}
        for control in ['gfnet_matched','lowrank16']:
            for fraction in FRACTIONS:
                estimates=[];distributions=[]
                for seed in SEEDS:
                    a,da=scores[('psan_ad0',seed,fraction)];b,db=scores[(control,seed,fraction)]
                    estimates.append(a-b);distributions.append(da-db)
                    differences[(control,seed,fraction)]=(a-b,da-db)
                simple.append(dict(dataset=dataset,control=control,fraction=fraction,
                                   **summarize(estimates,distributions,picks)))
        for row,adjusted in zip(simple,holm([r['p_uncorrected'] for r in simple])):
            row.update(p_holm=float(adjusted),family_size=6)
        simple_rows.extend(simple)
        interactions=[]
        for control in ['gfnet_matched','lowrank16']:
            for fraction in [10,25]:
                estimates=[];distributions=[]
                for seed in SEEDS:
                    a,da=differences[(control,seed,fraction)];b,db=differences[(control,seed,100)]
                    estimates.append(a-b);distributions.append(da-db)
                interactions.append(dict(dataset=dataset,control=control,fraction=fraction,reference_fraction=100,
                                         **summarize(estimates,distributions,picks)))
        for row,adjusted in zip(interactions,holm([r['p_uncorrected'] for r in interactions])):
            row.update(p_holm=float(adjusted),family_size=4)
        interaction_rows.extend(interactions)
    tables=OUT/'tables';tables.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(tables/'per_run.csv',index=False)
    pd.DataFrame(simple_rows).to_csv(tables/'contrasts.csv',index=False)
    pd.DataFrame(interaction_rows).to_csv(tables/'interactions.csv',index=False)
    for dataset in DATASETS:
        if spec['data'][dataset]['audit']['excluded_test_images']==0:
            observed=pd.DataFrame(rows).query('dataset == @dataset')
            if not np.allclose(observed.primary_macro_f1,observed.retained_macro_f1,atol=1e-12):
                raise ValueError('Zero-exclusion sensitivity changed the point estimates.')
    write_json(OUT/'analysis_provenance.json',dict(status='COMPLETED',completed_cells=54,
        protocol_sha256=sha256(PROTOCOL),primary_analysis_provenance_sha256=sha256(PRIMARY/'analysis_provenance.json'),
        no_classifier_outcome_selection=True,source_code_sha256=spec['source_code_sha256'],
        audit=spec['data'],tables={str(p.relative_to(ROOT)):sha256(p) for p in tables.glob('*.csv')}))
    print(pd.DataFrame(simple_rows).to_string(index=False),flush=True)
    print(pd.DataFrame(interaction_rows).to_string(index=False),flush=True)


def watch():
    verify()
    while not (PRIMARY/'analysis_provenance.json').exists():
        completed=len(list((PRIMARY/'final').glob('*/summary.json')))
        write_json(OUT/'completion_status.json',dict(state='waiting_for_primary',completed=completed,planned=54,
                   updated_kst=datetime.now(ZoneInfo('Asia/Seoul')).isoformat()))
        time.sleep(30)
    run()
    write_json(OUT/'completion_status.json',dict(state='complete',completed=54,planned=54,
               updated_kst=datetime.now(ZoneInfo('Asia/Seoul')).isoformat()))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['freeze','run','watch'])
    args=parser.parse_args()
    with threadpool_limits(limits=4):
        if args.command=='freeze': freeze()
        elif args.command=='run': run()
        else: watch()
