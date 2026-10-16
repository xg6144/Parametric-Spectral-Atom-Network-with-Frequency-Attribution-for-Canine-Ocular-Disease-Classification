"""Calibration, provenance-aware aggregation, controlled comparisons, errors."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax
from sklearn.metrics import (average_precision_score,confusion_matrix,f1_score,
                             precision_recall_curve,roc_auc_score)

from analyze_predictions import load_predictions
from evaluation_metrics import (cluster_bootstrap_ci,compute_metric_bundle,
                                paired_cluster_bootstrap,save_reliability_diagram)
from .common import ROOT,CLASSES,load_manifest,sha256,status,write_json

plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'ps.fonttype':42})


def savefig(fig,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path.with_suffix('.pdf'),bbox_inches='tight')
    fig.savefig(path.with_suffix('.png'),dpi=300,bbox_inches='tight')
    plt.close(fig)


def read_new(path):
    frame=pd.read_csv(path,keep_default_na=False)
    probs=frame.filter(regex=r'^prob::').to_numpy(dtype=float)
    logits=frame.filter(regex=r'^logit::').to_numpy(dtype=float)
    names=[p.split('::',1)[1] for p in frame if p.startswith('prob::')]
    if not np.allclose(probs.sum(1),1,atol=1e-5): raise ValueError('Invalid probabilities')
    if frame.path.duplicated().any(): raise ValueError('Duplicate prediction path')
    return frame,probs,logits,names


def quick_ci(labels,probs,groups,args):
    return cluster_bootstrap_ci(labels,probs,groups,iterations=args.bootstrap,seed=args.seed,
              metric_names=('macro_f1','ece','brier_score','nll'),n_bins=15)


def reliability_plots(labels,p,names,title,out):
    bundle=compute_metric_bundle(labels,p,names,n_bins=15)
    path=ROOT/'results/figures'/out
    save_reliability_diagram(bundle,path.with_suffix('.pdf'),title)
    save_reliability_diagram(bundle,path.with_suffix('.png'),title)
    bins=np.linspace(0,1,16);confidence=p.max(-1);correct=(p.argmax(-1)==labels)
    idx=np.clip(np.digitize(confidence,bins)-1,0,14)
    acc=np.array([correct[idx==b].mean() if (idx==b).any() else np.nan for b in range(15)])
    conf=np.array([confidence[idx==b].mean() if (idx==b).any() else np.nan for b in range(15)])
    fig,axes=plt.subplots(1,2,figsize=(9,3.5))
    axes[0].hist(confidence,bins=bins,color='#4477aa');axes[0].set(xlabel='Maximum class probability',ylabel='Images',title='Confidence histogram')
    axes[1].plot(conf,acc,'o-',color='#4477aa');axes[1].plot([0,1],[0,1],'--',color='gray')
    axes[1].set(xlim=(0,1),ylim=(0,1),xlabel='Mean confidence',ylabel='Empirical accuracy',title='Accuracy vs confidence')
    fig.suptitle(title);savefig(fig,path.parent/(out+'_confidence'))
    return bundle


def calibration(args):
    status('calibration','IN PROGRESS')
    rows=[];checks=[]
    for path in sorted((ROOT/'eval_results').glob('*_predictions.csv')):
        data=load_predictions(path)
        label,p,names=data['labels'],data['probabilities'],data['class_names']
        tag=path.name.replace('_predictions.csv','')
        bundle=reliability_plots(label,p,names,'Legacy: '+tag,'legacy_'+tag+'_reliability')
        # Legacy patient/visual groups are unavailable; use explicitly image bootstrap.
        ci=quick_ci(label,p,[str(i) for i in range(len(label))],args)
        write_json(ROOT/'results'/('legacy_'+tag+'_calibration.json'),dict(bundle=bundle,ci=ci,
                   cohort='legacy_saved_test_probabilities',uncertainty_unit='image; dog/visit independence unknown',
                   temperature_scaling='BLOCKED: no matching validation outputs/split'))
        row=dict(cohort='legacy_saved_test',model=tag,n=len(label),temperature=None,**bundle['metrics'])
        for metric,values in ci['metrics'].items():
            row[metric+'_ci_low']=values['ci95_lower'];row[metric+'_ci_high']=values['ci95_upper']
        rows.append(row)
        original=path.with_name(tag+'_eval_metrics.json')
        if original.exists():
            saved=json.loads(original.read_text())['metrics']
            for metric in ['accuracy','macro_f1','macro_auc_ovr']:
                delta=abs(bundle['metrics'][metric]-saved[metric])
                checks.append(dict(model=tag,metric=metric,saved=saved[metric],recomputed=bundle['metrics'][metric],abs_difference=delta,agrees=delta<1e-6))
        print('Legacy calibration',tag,bundle['metrics']['ece'],flush=True)
    summary_paths=list(sorted((ROOT/'results/training').glob('*/summary.json')))
    summary_paths+=list(sorted((ROOT/'results/tuning/final_selected').glob('*.json')))
    for summary_path in summary_paths:
        summary=json.loads(summary_path.read_text());directory=(ROOT/summary['checkpoint']).parent
        tuned=summary_path.parent.name=='final_selected'
        vf,vp,vl,names=read_new(directory/'val_predictions.csv')
        tf,tp,tl,_=read_new(directory/'test_predictions.csv')
        if len(vl)==0 or len(tl)==0: raise ValueError('Temperature scaling needs actual logits')
        if set(vf.path)&set(tf.path): raise ValueError('Validation/test paths overlap')
        # Optimize strictly on VALIDATION; final TEST never enters the objective.
        objective=lambda logt: float(-log_softmax(vl/np.exp(logt),axis=1)[np.arange(len(vf)),vf.label.to_numpy(dtype=int)].mean())
        fit=minimize_scalar(objective,bounds=(-4,4),method='bounded')
        if not fit.success: raise RuntimeError('Temperature fit failed')
        temperature=float(np.exp(fit.x));scaled=softmax(tl/temperature,axis=1)
        assert np.array_equal(scaled.argmax(-1),tp.argmax(-1))
        tag=summary['tag'];y=tf.label.to_numpy(dtype=int)
        validation_hash=sha256(directory/'val_predictions.csv')
        test_hash=sha256(directory/'test_predictions.csv')
        for condition,p in [('raw',tp),('temperature',scaled)]:
            cached_path=directory/('calibration_'+condition+'.json')
            cached=json.loads(cached_path.read_text()) if cached_path.exists() else {}
            valid=(cached.get('validation_prediction_sha256')==validation_hash and
                   cached.get('test_prediction_sha256')==test_hash and
                   cached.get('ci',{}).get('iterations')==args.bootstrap and
                   cached.get('ci',{}).get('seed')==args.seed and
                   abs(cached.get('temperature',0)-temperature)<1e-10 and
                   (ROOT/'results/figures'/(tag+'_'+condition+'_reliability.pdf')).exists())
            if valid:
                bundle,ci=cached['bundle'],cached['ci']
            else:
                bundle=reliability_plots(y,p,names,tag+' '+condition,tag+'_'+condition+'_reliability')
                ci=quick_ci(y,p,tf.group_id.to_numpy(),args)
            write_json(directory/('calibration_'+condition+'.json'),dict(bundle=bundle,ci=ci,temperature=temperature,
                fit_split='validation',fit_validation_nll=fit.fun,validation_prediction_sha256=validation_hash,
                test_prediction_sha256=test_hash,
                uncertainty_unit='visual group' if summary['dataset']=='canine' else
                    'lesion (published corrected mapping); patient unknown' if summary['dataset'].startswith('dermamnist_c') else 'image; identity unknown'))
            row=dict(cohort=summary['dataset']+('_tuned_test' if tuned else '_fresh_test'),model=tag,n=len(y),
                     condition=condition,temperature=temperature if condition=='temperature' else 1.,**bundle['metrics'])
            for metric,values in ci['metrics'].items():
                row[metric+'_ci_low']=values['ci95_lower'];row[metric+'_ci_high']=values['ci95_upper']
            rows.append(row)
        pd.DataFrame({'class':names,
            'raw_ece':[v['ece_ovr'] for v in compute_metric_bundle(y,tp,names)['per_class']],
            'temperature_ece':[v['ece_ovr'] for v in compute_metric_bundle(y,scaled,names)['per_class']]}).to_csv(directory/'classwise_calibration.csv',index=False)
        print('Validation-only temperature',tag,temperature,flush=True)
    pd.DataFrame(rows).to_csv(ROOT/'results/tables/calibration.csv',index=False)
    pd.DataFrame(checks).to_csv(ROOT/'results/tables/legacy_metric_verification.csv',index=False)
    assert all(r['agrees'] for r in checks), 'Saved metric mismatch'
    legacy_table=pd.DataFrame(rows).query('cohort == "legacy_saved_test"')[['model','n','macro_f1','macro_auc_ovr','ece','brier_score','nll']]
    (ROOT/'reports/02_calibration.md').write_text('''# Calibration reevaluation

Status: COMPLETED for stored probability cohorts and all completed fresh runs; AFNO historical calibration BLOCKED (no probabilities/checkpoint). A stored-cohort calculation is an independent metric reanalysis, not an independent retraining reproduction.

Top-label ECE uses 15 equal-width bins: sum_b n_b/N * |mean confidence_b - empirical accuracy_b|. Empty bins contribute zero. Classwise ECE is the mean over one-vs-rest class probabilities with the same 15 bins. Multiclass Brier = mean_i sum_c (p_ic - 1[y_i=c])^2 (not divided by C). NLL = -mean log(p_i,y_i), with the shared metric implementation's numerical clipping. CIs use 1,000 percentile bootstrap resamples at the available independence unit (images for legacy; visual groups for canine revision). These CIs describe test-sample uncertainty conditional on the trained checkpoint, not seed uncertainty. Reliability PDF/300-dpi PNG and confidence plots are reproduced directly from CSV.

Temperature is fitted by validation NLL only, bounded positive T=exp(t), t in [-4,4]. Both validation and test logits are saved; checks enforce disjoint paths. Temperature scaling preserves all argmax predictions and therefore Macro F1. Legacy temperature scaling is BLOCKED because matching validation outputs/split are unavailable; it is never fitted to test. The evaluated device/visual groups do not establish patient independence.

Legacy results (superseded split; do not mix with fresh results):

'''+legacy_table.to_csv(index=False)+'\nAll actual outputs and bootstrap endpoints are in `results/tables/calibration.csv`. Raw vs calibrated classwise ECE is saved next to each completed run. References: [Guo et al., ICML 2017](https://proceedings.mlr.press/v70/guo17a.html).\n')
    status('calibration','COMPLETED',legacy_models=len(legacy_table),fresh_conditions=sum(r['cohort']!='legacy_saved_test' for r in rows))


def aggregate_new_runs(args):
    summaries=[json.loads(p.read_text()) for p in sorted((ROOT/'results/training').glob('*/summary.json'))]
    rows=[]
    for s in summaries:
        config=json.loads((ROOT/s['checkpoint']).with_name('config.json').read_text())
        factors={k:config[k] for k in ['atoms','init','axis_aligned','fixed_phase','img_size','epochs']}
        rows.append({k:v for k,v in s.items() if k not in ['test','val']} | factors | {'validation_macro_f1':s['val']['macro_f1']} | s['test'])
    if not rows:
        status(args.phase,'NOT STARTED',reason='No completed fresh training summary')
        return
    frame=pd.DataFrame(rows);frame.to_csv(ROOT/'results/tables/fresh_per_run.csv',index=False)
    metrics=['macro_f1','macro_auc_ovr','accuracy','balanced_accuracy','ece','brier_score','nll','training_time_sec']
    aggregates=[]
    axes=['dataset','model','atom','atom_dropout','pretrained','atoms','init','axis_aligned','fixed_phase','img_size','epochs']
    for keys,part in frame.groupby(axes,dropna=False):
        row=dict(zip(axes,keys));row['seeds']=','.join(str(s) for s in sorted(part.seed));row['runs']=len(part)
        for metric in metrics:
            row[metric+'_mean']=part[metric].mean();row[metric+'_sd']=part[metric].std(ddof=1) if len(part)>1 else np.nan
        aggregates.append(row)
    pd.DataFrame(aggregates).to_csv(ROOT/'results/tables/fresh_aggregate.csv',index=False)
    legacy=[]
    for p in sorted((ROOT/'multi_seed_results').glob('*_predictions.csv')):
        data=load_predictions(p);bundle=compute_metric_bundle(data['labels'],data['probabilities'],data['class_names'])
        legacy.append({'model_seed':p.name.replace('_predictions.csv',''),'n':len(data['labels']),**bundle['metrics']})
    pd.DataFrame(legacy).to_csv(ROOT/'results/tables/legacy_seed_recomputed.csv',index=False)
    canine=[s for s in summaries if s['dataset']=='canine' and s['model'] in ['psan','gfnet','afno']
            and (s['model']!='psan' or s['tag'].startswith('canine_psan_gabor_ad0p2_M16_morlet_s'))]
    counts={m:len([s for s in canine if s['model']==m]) for m in ['psan','gfnet','afno']}
    complete=all(v>=3 for v in counts.values())
    five=all({42,43,44,45,46}<={s['seed'] for s in canine if s['model']==m} for m in counts)
    (ROOT/'reports/04_multiseed_reproducibility.md').write_text(f'''# Fresh multi-seed reproducibility

Status: {'COMPLETED (five seeds per core model)' if five else 'COMPLETED for three-seed scope; follow-up in progress' if complete else 'IN PROGRESS / partial results only'}.

Actual completed core runs: {counts}. Five-seed completion: {five}. Seeds42–46 are reported only when actually completed. New model selection uses validation fold2 only; fold1 is used for final evaluation after checkpoint selection. Folds3–5=train. Visual groups and byte hashes are disjoint, but dog identities are unavailable. All runs start from scratch; original group-CV checkpoints are not used to initialize fresh training. Tuned seed42 follow-up results are separate from this fixed-preset repeated training.

Thirty epochs maximum, patience10, warmup3, batch128, fp32, lr1e-4, decay1e-3, gradient clipping1, EMA=.999 from epoch1, inverse-square-root class weighted replacement sampler, original shared resize/flip/color augmentation and ImageNet normalization. One frozen configuration per arm, with no test-directed tuning. This is an equal-budget study with a shorter schedule than the original 300-epoch upper bound; it does not prove full convergence. AFNO's old model-specific lr/decay differ from the new common preset.

All per-run best epoch/validation F1/test metrics/training time/checkpoint/config fingerprints are saved under `results/training/`; `fresh_per_run.csv` and `fresh_aggregate.csv` contain the corresponding table. Mean ± SAMPLE SD across seeds measures training randomness conditional on one split. Bootstrap intervals measure held-out sample variability conditional on a checkpoint. Neither covers unknown patient dependence. Historical 42/123/2024 six-baseline results are separately recalculated in `legacy_seed_recomputed.csv`; no old PSAN/AFNO multi-seed evidence is available.
''')
    status('multiseed','COMPLETED' if five else 'IN PROGRESS',completed_core_runs=counts,seeds_requested=5,five_seeds_complete=five)


def ablation(args):
    # The same routine owns all correction families and paired cell alignment.
    from .statistics import run as controlled_statistics
    controlled_statistics(args)
    effects=pd.read_csv(ROOT/'results/tables/factorial_joint_effects.csv')
    complete_seeds=sorted(effects.seed.unique().tolist())
    status('ablation','COMPLETED' if len(complete_seeds)==3 else 'IN PROGRESS',
           scope='controlled single-seed OFAT; repeated factorial',
           complete_factorial_seeds=complete_seeds,legacy_ablation_predictions_missing=True,
           other_ofat_seed_replication='NOT STARTED')


def classwise(args):
    rows=[];device_rows=[]
    for path in sorted((ROOT/'results/training').glob('*/summary.json')):
        s=json.loads(path.read_text())
        if s['dataset']!='canine':continue
        directory=path.parent;frame,p,_,names=read_new(directory/'test_predictions.csv');y=frame.label.to_numpy(dtype=int)
        bundle=compute_metric_bundle(y,p,names)
        for r in bundle['per_class']:rows.append({'model_seed':s['tag'],**r})
        if s['seed']!=42:continue
        fig,ax=plt.subplots(figsize=(7.2,6.7));cm=np.asarray(bundle['confusion_matrix_normalized'])
        im=ax.imshow(cm,vmin=0,vmax=1,cmap='Blues');labels=[n.replace('_',' ') for n in names]
        ax.set_xticks(range(len(names)),labels=labels,rotation=60,ha='right',fontsize=9)
        ax.set_yticks(range(len(names)),labels=labels,fontsize=9)
        for i in range(len(names)):
            for j in range(len(names)):ax.text(j,i,f'{cm[i,j]:.2f}',ha='center',va='center',fontsize=8,color='white' if cm[i,j]>.5 else 'black')
        ax.set(xlabel='Predicted class',ylabel='True class',title=s['tag']);fig.colorbar(im,ax=ax,label='Row-normalized proportion');savefig(fig,ROOT/'results/figures'/(s['tag']+'_confusion'))
        fig,ax=plt.subplots(figsize=(6.5,4.7))
        for c in ['Nuclear_sclerosis','Eyelid_tumor','Non_ulcerative_keratitis','Pigmentary_keratitis','Epiphora']:
            i=names.index(c);precision,recall,_=precision_recall_curve(y==i,p[:,i]);ap=average_precision_score(y==i,p[:,i])
            ax.plot(recall,precision,label=c.replace('_',' ')+f' (AP={ap:.3f})')
        ax.set(xlabel='Recall',ylabel='Precision',xlim=(0,1),ylim=(0,1));ax.legend(fontsize=8);savefig(fig,ROOT/'results/figures'/(s['tag']+'_minority_pr'))
        # Explicitly exclude unsupported true classes from subgroup macro averaging.
        for device,part in frame.groupby('device'):
            ii=part.index.to_numpy();supported=np.unique(y[ii]);pred=p[ii].argmax(-1)
            values=dict(tag=s['tag'],device=device,images=len(ii),supported_classes='|'.join(names[c] for c in supported),
                supported_class_count=len(supported),accuracy=float((pred==y[ii]).mean()),
                macro_f1_supported=float(f1_score(y[ii],pred,labels=supported,average='macro',zero_division=0)))
            device_rows.append(values)
            matrix=confusion_matrix(y[ii],pred,labels=range(len(names)))
            pd.DataFrame(matrix,index=names,columns=names).to_csv(ROOT/'results/tables'/(s['tag']+'_device_'+str(len(device_rows))+'_confusion.csv'))
        # Cross-device common TRUE classes, retaining full-label model outputs.
        common=set(range(len(names)))
        for _,part in frame.groupby('device'):common&=set(part.label)
        for device,part in frame.groupby('device'):
            part=part[part.label.isin(common)];ii=part.index.to_numpy()
            if len(ii):device_rows.append(dict(tag=s['tag'],device=device,images=len(ii),
                supported_classes='|'.join(names[c] for c in sorted(common)),supported_class_count=len(common),
                protocol='common_true_classes_full_classifier',accuracy=float((p[ii].argmax(-1)==y[ii]).mean()),
                macro_f1_supported=float(f1_score(y[ii],p[ii].argmax(-1),labels=sorted(common),average='macro',zero_division=0))))
    if rows:pd.DataFrame(rows).to_csv(ROOT/'results/tables/fresh_classwise.csv',index=False)
    if device_rows:pd.DataFrame(device_rows).to_csv(ROOT/'results/tables/device_performance.csv',index=False)
    tuned_rows=[]
    for path in sorted((ROOT/'results/tuning/final_selected').glob('*.json')):
        summary=json.loads(path.read_text());directory=(ROOT/summary['checkpoint']).parent
        frame,p,_,names=read_new(directory/'test_predictions.csv')
        bundle=compute_metric_bundle(frame.label.to_numpy(dtype=int),p,names)
        for r in bundle['per_class']:
            tuned_rows.append(dict(model=summary['model'],model_seed=summary['tag'],
                checkpoint_sha256=summary['checkpoint_sha256'],cohort='canine_tuned_test',**r))
    if tuned_rows:pd.DataFrame(tuned_rows).to_csv(ROOT/'results/tables/validation_tuned_classwise.csv',index=False)
    (ROOT/'reports/11_classwise_error_analysis.md').write_text(f'''# Minority class and acquisition error analysis

Status: {'COMPLETED for available fresh prediction metrics/PR/confusion/device patterns' if rows else 'NOT STARTED: fresh predictions needed'}; real image explanation examples are tracked in the attribution artifacts.

Per-class precision/recall/F1/AP/ROC-AUC/ECE/support in `fresh_classwise.csv` ({len(rows)} rows). All eleven classes use the same alphabetical index mapping. Seed42 confusion matrices are separate legible vector PDFs plus 300-dpi PNG, with numeric row proportions, axis labels and colorbar. PR curves focus on the five reviewer-highlighted classes and report average precision. No invented clinical explanation or synthetic eye photograph is used.

Device subgroup tables contain true class coverage and sample sizes. Macro F1 is averaged over supported true classes only; therefore it is not directly comparable to the eleven-class pooled F1. A common-supported-true-class analysis retains all model output labels, counting predictions outside those classes as errors. Device-specific count confusion CSVs include all output classes so unsupported predictions remain visible. This analysis describes association conditional on the frozen split, not causal elimination of capture-device confounding. Dog/hospital independence remains unknown.
''')
    status('classwise','COMPLETED' if rows else 'NOT STARTED',metric_rows=len(rows),device_rows=len(device_rows))


def run(args):
    if args.phase=='calibration':calibration(args)
    elif args.phase=='ablation':
        if args.train:
            from .training import run as training_run
            for atom in ['gabor','gaussian']:
                for drop in [0.,.2]:
                    args.atom=atom;args.atom_dropout=drop;args.models=['psan'];training_run(args)
        ablation(args)
        from .statistics import run as statistical_run
        statistical_run(args)
    elif args.phase=='classwise':
        classwise(args)
        from .supplementary import device_recall
        device_recall()
    else:aggregate_new_runs(args)
