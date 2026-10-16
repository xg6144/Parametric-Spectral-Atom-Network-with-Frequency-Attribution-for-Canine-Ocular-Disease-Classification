"""Unified CLI adapter and source-linked tables for the frozen tuning study."""
import json
import subprocess
import sys

import pandas as pd

from .common import ROOT, sha256, status, write_json


def aggregate():
    protocol = json.loads((ROOT/'configs/validation_tuning.json').read_text())
    trials=[]; selected=[]
    for p in sorted((ROOT/'results/tuning/trials').glob('*/summary.json')):
        s=json.loads(p.read_text());cfg=json.loads(p.with_name('config.json').read_text())
        if 'test' in s or any(p.parent.glob('test*')):
            raise ValueError('Screening directory contains test evidence')
        trials.append({k:s[k] for k in ['model','candidate','seed','pretrained','best_validation_epoch',
                      'best_validation_macro_f1','epochs_completed','training_time_sec','checkpoint_sha256']} |
                      dict(lr=cfg['lr'],weight_decay=cfg['weight_decay'],batch_size=cfg['batch_size'],
                           protocol_sha256=s['protocol_sha256']))
    for p in sorted((ROOT/'results/tuning/final_selected').glob('*.json')):
        s=json.loads(p.read_text());selection=json.loads((ROOT/'results/tuning/selections'/(p.stem+'.json')).read_text())
        cfg=json.loads((ROOT/s['checkpoint']).with_name('config.json').read_text())
        if selection['test_used']:
            raise ValueError('Selection used test')
        selected.append(dict(model=p.stem,seed=s['seed'],pretrained=s['pretrained'],candidate=s['candidate'],
            lr=cfg['lr'],weight_decay=cfg['weight_decay'],screening_best_validation_macro_f1=max(selection['validation_scores']),
            best_validation_epoch=s['best_validation_epoch'],validation_macro_f1=s['val']['macro_f1'],
            reused_completed_run=s.get('reused_completed_run',False),training_time_sec=s['training_time_sec'],
            checkpoint=s['checkpoint'],checkpoint_sha256=s['checkpoint_sha256'],
            split_sha256=s['split_sha256'],**s['test']))
    tables=ROOT/'results/tables'
    if trials:pd.DataFrame(trials).to_csv(tables/'validation_tuning_trials.csv',index=False)
    if selected:pd.DataFrame(selected).to_csv(tables/'validation_tuned_final.csv',index=False)
    complete=len(trials)==24 and len(selected)==8
    equal = len(trials)==24 and all(
        len([r for r in trials if r['model']==m])==3 and
        all(r['epochs_completed']==8 and r['batch_size']==128 for r in trials if r['model']==m)
        for m in protocol['models'])
    if len(trials)==24 and not equal:
        raise ValueError('Completed tuning budget differs between arms')
    write_json(ROOT/'results/tuning/aggregate_provenance.json',dict(
        protocol_sha256=sha256(ROOT/'configs/validation_tuning.json'),
        completed_trials=len(trials),completed_final_arms=len(selected),equal_screening_budget_verified=equal,
        screening_test_artifacts=False,prior_holdout_exposure=True,
        inference='bounded short-schedule validation screen; not optimal tuning or a new blinded holdout'))
    status('tuning','COMPLETED' if complete else 'IN PROGRESS',trials=len(trials),final_arms=len(selected),
           equal_screening_budget_verified=equal)


def run(args):
    if args.train:
        workers=[args.tuning_worker] if args.tuning_worker is not None else [0,1]
        for worker in workers:
            subprocess.run([sys.executable,'-m','experiments.validation_tuning','--worker',str(worker),
                            '--device',args.device],cwd=ROOT,check=True)
    aggregate()


if __name__=='__main__':aggregate()
