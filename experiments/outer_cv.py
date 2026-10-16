"""Five outer visual-group folds with a separate inner validation partition.

Fold1 reuses the identical completed common-preset seed42 runs. Other folds
run the byte-preserved shared trainer in isolated local workspaces.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from evaluation_metrics import compute_metric_bundle
from .analysis import read_new
from .common import ROOT,load_manifest,sha256,status,write_json
from .reproduction_workspace import create


def setup():
    base=load_manifest();base_sha=sha256(ROOT/'results/revision_split.csv')
    protocol=dict(protocol='five_outer_visual_group_folds_with_separate_validation_v1',
        folds=[dict(test=i,validation=i%5+1,train=[j for j in range(1,6) if j not in [i,i%5+1]]) for i in range(1,6)],
        models=['psan','gfnet','afno'],seed=42,config='common c0: LR1e-4 WD1e-3 batch128 fp32 max30 patience10',
        base_manifest_sha256=base_sha,settings='same preset fixed before original main-test outcomes; no outer-test hyperparameter selection',
        fold1='reuse verified identical completed main seed42 checkpoints without retraining',
        uncertainty='fold SD is descriptive split variation; not seed SD or independent patient validation',
        previous_corpus_exposure=True,patient_independence_established=False,
        frozen_source_manifest_sha256=sha256(ROOT/'revision_history/20261013_round2_training_source/manifest.json'))
    path=ROOT/'configs/outer_cv.json'
    if path.exists() and json.loads(path.read_text())!=protocol:raise ValueError('Frozen outer CV protocol changed')
    write_json(path,protocol)
    for fold in protocol['folds'][1:]:
        i=fold['test'];dest=ROOT/'results/cross_validation'/f'fold{i}'/'project'
        if dest.exists():continue
        create(dest)
        frame=base.copy();frame['split']=np.where(frame.fold==i,'test',np.where(frame.fold==fold['validation'],'val','train'))
        if (frame.groupby('group_id').split.nunique()>1).any():raise ValueError('CV group overlap')
        manifest=dest/'results/revision_split.csv';proof_path=dest/'results/corrected_integrity.json'
        # Detach only the newly created references before writing local CV inputs.
        for p,original in [(manifest,ROOT/'results/revision_split.csv'),(proof_path,ROOT/'results/corrected_integrity.json')]:
            if not p.is_symlink() or p.resolve()!=original.resolve():raise ValueError('Unexpected CV input reference')
            p.unlink()
        frame.to_csv(manifest,index=False)
        proof=json.loads((ROOT/'results/corrected_integrity.json').read_text())
        proof['revision_manifest_sha256']=sha256(manifest);proof['split_counts']=frame.split.value_counts().to_dict()
        proof['outer_cv']=fold;proof.pop('device_only_metrics',None);write_json(proof_path,proof)
        cfg=json.loads((ROOT/'configs/revision.json').read_text())
        cfg['revision_protocol']=f'outer_visual_group_test{i}_validation{fold["validation"]}_v1'
        cfg['test_fold']=i;cfg['validation_fold']=fold['validation'];cfg['train_folds']=fold['train'];cfg['seeds']=[42]
        cfg['notes']='Frozen common preset; separate outer test/inner validation; no test-directed tuning; prior corpus exposure disclosed'
        write_json(dest/'configs/outer_cv_run.json',cfg)
        write_json(dest/'OUTER_CV_PROVENANCE.json',dict(fold,protocol_sha256=sha256(path),manifest_sha256=sha256(manifest)))
    if sha256(ROOT/'results/revision_split.csv')!=base_sha:raise ValueError('Global frozen split changed')
    print('Outer CV setup verified; original split unchanged',flush=True)


def child_train(device):
    import torch
    from .training import train_one
    torch.set_num_threads(4)
    args=SimpleNamespace(phase='multiseed',dataset='canine',config=str(ROOT/'configs/outer_cv_run.json'),
        epochs=None,resume=True,device=device,atom='gabor',atom_dropout=.2,atoms=16,
        init='morlet',axis_aligned=False,fixed_phase=False)
    for name in ['psan','gfnet','afno']:
        train_one(args,name,42);torch.cuda.empty_cache()


def worker(fold,device):
    dest=ROOT/'results/cross_validation'/f'fold{fold}'/'project'
    if not dest.exists():raise ValueError('Run --setup before CV workers')
    log=ROOT/'results/experiment_logs'/f'outer_cv_fold{fold}.log'
    # The standalone trainer source is copied at setup and remains byte preserved.
    command=[sys.executable,'-c','from experiments.outer_cv import child_train; import sys; child_train(sys.argv[1])',device]
    with log.open('a') as stream:subprocess.run(command,cwd=dest,stdout=stream,stderr=subprocess.STDOUT,check=True)
    status('outer_cv_fold'+str(fold),'COMPLETED',log=str(log.relative_to(ROOT)))


def aggregate():
    protocol=json.loads((ROOT/'configs/outer_cv.json').read_text());rows=[];frames={m:[] for m in protocol['models']}
    for fold in protocol['folds']:
        i=fold['test'];root=ROOT if i==1 else ROOT/'results/cross_validation'/f'fold{i}'/'project'
        for model in protocol['models']:
            suffix='_gabor_ad0p2_M16_morlet' if model=='psan' else ''
            directory=root/'results/training'/f'canine_{model}{suffix}_s42'
            if not (directory/'summary.json').exists():continue
            s=json.loads((directory/'summary.json').read_text());f,p,_,names=read_new(directory/'test_predictions.csv')
            actual=compute_metric_bundle(f.label.to_numpy(),p,names)
            if abs(actual['metrics']['macro_f1']-s['test']['macro_f1'])>1e-7:raise ValueError('CV saved metric mismatch')
            if not f.fold.eq(i).all():raise ValueError('CV outer test contains another fold')
            frames[model].append(f)
            rows.append(dict(model=model,outer_fold=i,validation_fold=fold['validation'],seed=42,
                reused_main=(i==1),images=len(f),best_validation_epoch=s['best_validation_epoch'],
                validation_macro_f1=s['val']['macro_f1'],training_time_sec=s['training_time_sec'],
                checkpoint=str((root/s['checkpoint']).relative_to(ROOT)),checkpoint_sha256=s['checkpoint_sha256'],
                prediction=str((directory/'test_predictions.csv').relative_to(ROOT)),
                manifest_sha256=s['split_sha256'],**actual['metrics']))
    if not rows:return
    table=pd.DataFrame(rows);table.to_csv(ROOT/'results/tables/outer_cv_per_fold.csv',index=False)
    summaries=[];classes=[]
    for model,part in table.groupby('model'):
        row=dict(model=model,folds=len(part),seed=42)
        for key in ['macro_f1','macro_auc_ovr','accuracy','balanced_accuracy']:
            row[key+'_mean']=float(part[key].mean());row[key+'_sd']=float(part[key].std(ddof=1)) if len(part)>1 else None
        if len(part)==5:
            oof=pd.concat(frames[model],ignore_index=True).sort_values('path')
            base=load_manifest().sort_values('path')
            if not np.array_equal(oof.path,base.path) or not np.array_equal(oof.label,base.label):raise ValueError('OOF coverage mismatch')
            oof.to_csv(ROOT/'results/cross_validation'/(model+'_out_of_fold_predictions.csv'),index=False)
            p=oof.filter(regex=r'^prob::').to_numpy();names=[c.split('::',1)[1] for c in oof if c.startswith('prob::')]
            bundle=compute_metric_bundle(oof.label.to_numpy(),p,names)
            write_json(ROOT/'results/cross_validation'/(model+'_out_of_fold_metrics.json'),bundle)
            row['pooled_oof_macro_f1']=bundle['metrics']['macro_f1'];row['pooled_oof_macro_auc']=bundle['metrics']['macro_auc_ovr']
            for c in bundle['per_class']:classes.append(dict(model=model,**c))
        summaries.append(row)
    pd.DataFrame(summaries).to_csv(ROOT/'results/tables/outer_cv_aggregate.csv',index=False)
    if classes:pd.DataFrame(classes).to_csv(ROOT/'results/tables/outer_cv_classwise.csv',index=False)
    status('outer_cv','COMPLETED' if len(rows)==15 else 'IN PROGRESS',completed_folds_models=len(rows),
           newly_trained=len([r for r in rows if not r['reused_main']]),fold1_reused=3,patient_independence=False)


def run(args):
    if args.train:
        setup()
        for fold in ([args.cv_fold] if args.cv_fold is not None else [2,3,4,5]):
            worker(fold,args.device)
    aggregate()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--setup',action='store_true')
    parser.add_argument('--fold',type=int,choices=[2,3,4,5]);parser.add_argument('--device',default='cpu')
    parser.add_argument('--aggregate',action='store_true');args=parser.parse_args()
    if args.setup:setup()
    elif args.fold:worker(args.fold,args.device)
    else:aggregate()
