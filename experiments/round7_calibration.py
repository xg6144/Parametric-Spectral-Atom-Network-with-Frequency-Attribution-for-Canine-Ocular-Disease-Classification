"""Reviewer-driven vector recalibration, with validation-only regularization selection.

No classifier is retrained. This is a post hoc, within-corpus study, not a new
clinical validation. Positive diagonal vector scaling follows the multiclass
form in Guo et al. (2017), with a penalty shrinking deviations from one scale.
Run --freeze before --run; completed input/code hashes must remain unchanged.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.special import logsumexp, softmax
from sklearn.model_selection import StratifiedGroupKFold

from .common import ROOT, sha256, write_json
from evaluation_metrics import compute_metric_bundle

OUT = ROOT / 'results/round7/calibration'
PROTOCOL = ROOT / 'configs/round7_calibration_protocol.json'
SEEDS = [42, 43, 44]
MODELS = ['psan', 'gfnet_matched']
LAMBDAS = [0., 0.0001, 0.001, 0.01, 0.1, 1.]


def directory(model, seed):
    base = 'results/tuning/final' if model == 'psan' and seed == 42 else 'results/round3/tuning/final'
    return ROOT / base / f'canine_{model}_c2_s{seed}_final'


def arrays(frame):
    columns = [c for c in frame if c.startswith('logit::')]
    return frame[columns].to_numpy(dtype=np.float64), frame.label.to_numpy(dtype=np.int64), [c[7:] for c in columns]


def nll(logits, labels):
    return logsumexp(logits, axis=1) - logits[np.arange(len(labels)), labels]


def fit_temperature(logits, labels):
    result = minimize_scalar(lambda t: float(nll(logits / np.exp(t), labels).mean()),
                             bounds=(-5., 5.), method='bounded', options={'xatol': 1e-10})
    if not result.success:
        raise RuntimeError(f'Temperature fitting failed: {result.message}')
    return float(np.exp(result.x))


def fit_vector(logits, labels, strength):
    """Convex NLL in positive diagonal scales and biases, plus convex penalty.

    Shrink scale deviations from their mean and all biases; leave the common
    scale unpenalized. The scalar-temperature family is therefore nested.
    """
    count, classes = logits.shape
    scalar = 1. / fit_temperature(logits, labels)
    initial = np.r_[np.full(classes, scalar), np.zeros(classes)]

    def objective(params):
        scale, bias = params[:classes], params[classes:]
        transformed = logits * scale + bias
        probabilities = softmax(transformed, axis=1)
        loss = float(nll(transformed, labels).mean())
        probabilities[np.arange(count), labels] -= 1.
        delta = scale - scale.mean()
        penalty = strength * (np.mean(delta**2) + np.mean(bias**2))
        gradient = np.r_[(probabilities * logits).mean(axis=0) + 2*strength*delta/classes,
                         probabilities.mean(axis=0) + 2*strength*bias/classes]
        return loss + penalty, gradient

    result = minimize(objective, initial, jac=True, method='L-BFGS-B',
                      bounds=[(1e-4, 20.)]*classes + [(None, None)]*classes,
                      options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50})
    if not result.success:
        raise RuntimeError(f'Vector fitting failed: {result.message}')
    scale, bias = result.x[:classes], result.x[classes:]
    bias = bias - bias.mean()
    return dict(scale=scale.tolist(), bias=bias.tolist(), regularization=float(strength),
                converged=True, iterations=int(result.nit), objective=float(result.fun),
                gradient_max_abs=float(np.abs(result.jac).max()))


def transform(logits, params):
    return logits*np.asarray(params['scale']) + np.asarray(params['bias'])


def class_ece(labels, probabilities, bins=15):
    values = []
    for c in range(probabilities.shape[1]):
        confidence = probabilities[:, c]
        index = np.minimum((confidence*bins).astype(int), bins-1)
        counts = np.bincount(index, minlength=bins)
        positive = np.bincount(index, weights=(labels == c), minlength=bins)
        summed = np.bincount(index, weights=confidence, minlength=bins)
        values.append(float(np.abs(positive-summed).sum()/len(labels)))
    return values


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    if PROTOCOL.exists():
        raise FileExistsError('Frozen calibration protocol already exists; do not replace it.')
    reference = pd.read_csv(directory('psan', 42)/'val_predictions.csv')
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20261109)
    fit_idx, select_idx = next(splitter.split(reference, reference.label, reference.group_id))
    assignments = reference[['path', 'label', 'group_id']].copy()
    assignments['calibration_split'] = 'fit'
    assignments.loc[select_idx, 'calibration_split'] = 'selection'
    if (assignments.groupby('group_id').calibration_split.nunique() > 1).any():
        raise ValueError('Calibration visual groups overlap')
    assignments.to_csv(OUT/'validation_partition.csv', index=False)
    inputs = {}
    for model in MODELS:
        for seed in SEEDS:
            for filename in ['val_predictions.csv', 'test_predictions.csv', 'summary.json']:
                path = directory(model, seed)/filename
                inputs[str(path.relative_to(ROOT))] = sha256(path)
    write_json(PROTOCOL, dict(protocol='round7_positive_vector_scaling_v1', frozen_on='2026-11-09',
        post_hoc=True, prior_exposure='All submitted corpus/test findings were known; not a blinded holdout.',
        models=MODELS, seeds=SEEDS, regularization_candidates=LAMBDAS,
        fitting='NLL on four validation folds; positive diagonal scale plus bias; shrink scale deviations from mean and biases',
        selection='Lowest NLL on fifth visual-group-disjoint validation fold; ties favor stronger penalty; refit all validation',
        scalar='Positive global temperature, minimum full-validation NLL; unchanged classifier decisions',
        evaluation='Read test logits only after selection and refit for every model/seed; never select on test',
        metrics='NLL, Brier, Macro F1, top-label 15-bin ECE, mean one-vs-rest class ECE; 10/20 bins descriptive sensitivity',
        inference='B=10000 crossed visual-group x seed bootstrap; NLL/Brier/F1 vector-minus-temperature for two models, Holm6',
        classwise='Per-class ECE descriptive, no class-specific significance claim',
        bootstrap_seed=20261109, bootstrap_iterations=10000,
        validation_partition_sha256=sha256(OUT/'validation_partition.csv'),
        fit_n=len(fit_idx), selection_n=len(select_idx), inputs=inputs,
        sources={str(Path(p).relative_to(ROOT)):sha256(p) for p in [Path(__file__), ROOT/'evaluation_metrics.py', ROOT/'experiments/common.py']},
        primary_reference='https://proceedings.mlr.press/v70/guo17a.html'))
    print('Frozen', str(PROTOCOL.relative_to(ROOT)), flush=True)


def holm(values):
    values=np.asarray(values, dtype=float); order=np.argsort(values)
    adjusted=np.empty_like(values); adjusted[order]=np.minimum(1., np.maximum.accumulate(values[order]*np.arange(len(values),0,-1)))
    return adjusted


def bootstrap(model, saved):
    """Exact multinomial group bootstrap with crossed seed resampling.

    Each column is a group's contribution to losses/confusion, so the same
    group draw is paired across methods and seeds without a pixel-level IID
    assumption. Does not cover unknown patient dependence.
    """
    ref=saved[(model, SEEDS[0])]['frame']
    _, group=np.unique(ref.group_id.to_numpy(), return_inverse=True)
    group_n=group.max()+1; classes=saved[(model,SEEDS[0])]['probabilities']['vector'].shape[1]
    labels=ref.label.to_numpy(); metrics=['nll','brier_score','macro_f1']
    group_sizes=np.bincount(group, minlength=group_n).astype(float)
    stats=[]
    for seed in SEEDS:
        entry=saved[(model,seed)]
        if not np.array_equal(entry['frame'].path,ref.path): raise ValueError('Test rows differ between seeds')
        for method in ['temperature','vector']:
            probabilities=entry['probabilities'][method]
            losses=-np.log(np.maximum(probabilities[np.arange(len(labels)),labels],1e-300))
            brier=(probabilities**2).sum(1)-2*probabilities[np.arange(len(labels)),labels]+1
            confusion=np.zeros((group_n,classes*classes),dtype=np.float64)
            np.add.at(confusion,(group, labels*classes+probabilities.argmax(1)),1.)
            stats.append(np.column_stack([np.bincount(group,weights=losses,minlength=group_n),
                                          np.bincount(group,weights=brier,minlength=group_n),confusion]))
    width=2+classes*classes
    packed=np.concatenate([s.astype(np.float32) for s in stats],axis=1)
    rng=np.random.default_rng(20261109)
    draws=np.empty((10000,3),dtype=np.float64)
    for start in range(0,10000,32):
        size=min(32,10000-start)
        weights=np.empty((size,group_n),dtype=np.float32)
        for b in range(size): weights[b]=np.bincount(rng.integers(group_n,size=group_n),minlength=group_n)
        totals=(weights@packed).reshape(size,len(SEEDS),2,width).astype(np.float64)
        denom=(weights@group_sizes).reshape(size,1,1)
        loss=totals[:,:,:,:2]/denom[:,:,:,None]
        cm=totals[:,:,:,2:].reshape(size,len(SEEDS),2,classes,classes)
        diag=np.diagonal(cm,axis1=-2,axis2=-1)
        divisors=cm.sum(-1)+cm.sum(-2)
        f1=np.divide(2*diag,divisors,out=np.zeros_like(diag),where=divisors>0).mean(-1)
        all_metrics=np.concatenate([loss,f1[:,:,:,None]],axis=-1)
        differences=all_metrics[:,:,1,:]-all_metrics[:,:,0,:]
        seed_draw=rng.integers(len(SEEDS),size=(size,len(SEEDS)))
        draws[start:start+size]=np.take_along_axis(differences,seed_draw[:,:,None],axis=1).mean(1)
    rows=[]
    for j,metric in enumerate(metrics):
        actual=np.mean([saved[(model,s)]['metrics']['vector'][metric]-saved[(model,s)]['metrics']['temperature'][metric] for s in SEEDS])
        sample=draws[:,j]; lo,hi=np.quantile(sample,[.025,.975])
        p=min(1.,2*min((np.sum(sample<=0)+1)/10001,(np.sum(sample>=0)+1)/10001))
        rows.append(dict(model=model,metric=metric,comparison='vector minus temperature',mean_delta=actual,
                         ci_low=lo,ci_high=hi,p_uncorrected=p,bootstrap_iterations=10000))
    np.savez_compressed(OUT/f'{model}_crossed_bootstrap.npz',draws=draws,metrics=np.asarray(metrics))
    return rows


def run():
    protocol=json.loads(PROTOCOL.read_text())
    for relative,digest in {**protocol['inputs'],**protocol['sources']}.items():
        if sha256(ROOT/relative)!=digest: raise ValueError('Frozen input/source changed: '+relative)
    if sha256(OUT/'validation_partition.csv')!=protocol['validation_partition_sha256']: raise ValueError('Calibration partition changed')
    assignments=pd.read_csv(OUT/'validation_partition.csv')
    fit_mask=assignments.calibration_split.eq('fit').to_numpy()
    selections={}; trial_rows=[]
    # Complete ALL fitting/selection before reading ANY test logits.
    for model in MODELS:
        for seed in SEEDS:
            val=pd.read_csv(directory(model,seed)/'val_predictions.csv')
            if not np.array_equal(val.path,assignments.path) or not np.array_equal(val.label,assignments.label):
                raise ValueError('Validation alignment changed')
            logits,labels,classes=arrays(val)
            candidates=[]
            for strength in LAMBDAS:
                params=fit_vector(logits[fit_mask],labels[fit_mask],strength)
                score=float(nll(transform(logits[~fit_mask],params),labels[~fit_mask]).mean())
                trial_rows.append(dict(model=model,seed=seed,regularization=strength,selection_nll=score,iterations=params['iterations']))
                candidates.append((score,-strength,strength))
            selected=min(candidates)[2]
            full_params=fit_vector(logits,labels,selected)
            record=dict(model=model,seed=seed,classes=classes,temperature=fit_temperature(logits,labels),
                        vector=full_params,selection_nll=min(candidates)[0],selection_used_test=False)
            write_json(OUT/f'{model}_s{seed}_parameters.json',record)
            selections[(model,seed)]=record
            print('Selected calibration',model,seed,'lambda',selected,flush=True)
    pd.DataFrame(trial_rows).to_csv(OUT/'validation_trials.csv',index=False)
    rows=[];class_rows=[];saved={}
    for (model,seed),record in selections.items():
        frame=pd.read_csv(directory(model,seed)/'test_predictions.csv')
        logits,labels,classes=arrays(frame)
        transformed={'raw':logits,'temperature':logits/record['temperature'],'vector':transform(logits,record['vector'])}
        entries={};probabilities={}
        for method,z in transformed.items():
            p=softmax(z,axis=1); probabilities[method]=p
            metrics=compute_metric_bundle(labels,p,classes)['metrics']
            entries[method]=metrics
            rows.append(dict(model=model,seed=seed,method=method,
                             prediction_changes=int(np.sum(p.argmax(1)!=logits.argmax(1))),**metrics))
            for bins in [10,15,20]:
                for c,ece in enumerate(class_ece(labels,p,bins)):
                    class_rows.append(dict(model=model,seed=seed,method=method,bins=bins,
                                           class_name=classes[c],images=int(np.sum(labels==c)),ece=ece))
        if rows[-2]['prediction_changes'] != 0: raise ValueError('Global temperature changed argmax')
        np.savez_compressed(OUT/f'{model}_s{seed}_test_probabilities.npz',labels=labels,**probabilities)
        saved[(model,seed)]=dict(frame=frame[['path','group_id','label']],probabilities=probabilities,metrics=entries)
    table=pd.DataFrame(rows);table.to_csv(OUT/'metrics_per_run.csv',index=False)
    cw=pd.DataFrame(class_rows);cw.to_csv(OUT/'classwise_ece_per_run.csv',index=False)
    contrasts=[]
    for model in MODELS: contrasts.extend(bootstrap(model,saved))
    contrasts=pd.DataFrame(contrasts);contrasts['p_holm']=holm(contrasts.p_uncorrected)
    contrasts['family_size']=6;contrasts.to_csv(OUT/'contrasts.csv',index=False)
    columns=['macro_f1','accuracy','ece','classwise_ece','nll','brier_score','prediction_changes']
    summary=table.groupby(['model','method'])[columns].agg(['mean','std'])
    summary.columns=['_'.join(c) for c in summary.columns];summary.reset_index().to_csv(OUT/'summary.csv',index=False)
    write_json(OUT/'provenance.json',dict(status='COMPLETED',post_hoc=True,protocol_sha256=sha256(PROTOCOL),
        source_sha256=sha256(__file__),validation_partition_sha256=sha256(OUT/'validation_partition.csv'),
        models=MODELS,seeds=SEEDS,test_selection=False,
        scope='Within-corpus retrospective calibration; unknown patient dependence remains; group bootstrap conditional on fitted calibrators, not fit uncertainty.',
        artifacts={p.name:sha256(p) for p in OUT.glob('*') if p.is_file() and p.suffix in ['.csv','.npz','.json'] and p.name!='provenance.json'}))
    print(summary.to_string(),flush=True)
    print(contrasts.to_string(index=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--freeze',action='store_true');parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    if args.freeze: freeze()
    elif args.run: run()
    else: parser.error('Specify --freeze or --run')
