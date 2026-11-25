"""Post hoc completion of the 100-epoch atom-dropout comparison (2026-11-09).

Reuses experiments.round6.train unchanged and supplies an isolated frozen protocol
and output directory. Only the missing AD0 seeds 43 and 44 are trained. Existing
AD0 seed42 and AD0.2 seeds42--44 remain immutable source artifacts. GPU0 only.

    CUDA_VISIBLE_DEVICES=0 $PY -m experiments.round7_dropout freeze
    CUDA_VISIBLE_DEVICES=0 $PY -m experiments.round7_dropout replay
    CUDA_VISIBLE_DEVICES=0 $PY -m experiments.round7_dropout smoke
    CUDA_VISIBLE_DEVICES=0 $PY -m experiments.round7_dropout launch
    $PY -m experiments.round7_dropout status
    $PY -m experiments.round7_dropout analyze
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .common import CLASSES, ROOT, sha256, write_json
from . import round6 as training

OUT = ROOT / 'results/round7/dropout'
PROTOCOL = ROOT / 'configs/round7_dropout_protocol.json'
SEEDS = [42, 43, 44]
NEW_SEEDS = [43, 44]
SOURCE_FILES = sorted(set(training.SOURCES + ['experiments/round7_dropout.py',
                                              'experiments/round3_analysis.py']))
REFERENCE = {
    ('psan_ad0', 42): 'results/round4/long/canine_psan_ad0_c2_s42_long100',
    ('psan', 42): 'results/round4/long/canine_psan_c2_s42_long100',
    ('psan', 43): 'results/round6/long/canine_psan_c2_s43_long100',
    ('psan', 44): 'results/round6/long/canine_psan_c2_s44_long100',
}


def job(seed):
    return dict(exp='long', kind='final', arm='psan_ad0', candidate='c2', seed=seed)


def protocol_spec():
    refs = {}
    for (arm, seed), relative in sorted(REFERENCE.items()):
        directory = ROOT / relative
        summary = json.loads((directory / 'summary.json').read_text())
        if sha256(ROOT / summary['checkpoint']) != summary['checkpoint_sha256']:
            raise ValueError('Changed source checkpoint: ' + relative)
        refs[f'{arm}_{seed}'] = dict(directory=relative,
                                   config_sha256=sha256(directory / 'config.json'),
                                   summary_sha256=sha256(directory / 'summary.json'),
                                   checkpoint_sha256=summary['checkpoint_sha256'])
    return dict(
        protocol='round7_dropout_long_completion_v1', frozen_on='2026-11-09',
        post_hoc=True, not_a_blinded_holdout=True,
        motivation='Complete missing 100-epoch AD0 seeds to assess the practical dropout recommendation.',
        prior_exposure=['all submitted canine and MedMNIST+ results, including the single-seed 100-epoch '
                        'AD0.2 advantage and the 30-epoch dropout deficit'],
        new_runs=[job(s) for s in NEW_SEEDS], reference_runs=refs,
        split_sha256=sha256(ROOT / 'results/revision_split.csv'),
        arms={'psan_ad0': dict(name='psan', atoms=16, atom_dropout=0.)},
        common=dict(batch_size=128, num_workers=6, ema_decay=.999, gradient_clip=1., warmup_epochs=3,
                    stochastic_depth=.1, sampler='inverse-square-root class-weighted replacement sampling',
                    precision='float32 tensors; CUDA matmul TF32 off; cuDNN convolution TF32 default',
                    no_weight_decay=list(training.NO_DECAY)),
        training=dict(implementation='experiments.round6.train, unchanged', epochs=100,
                      learning_rate=.0005, weight_decay=.05, early_stopping=False,
                      selection='best validation Macro F1 checkpoint; final test evaluation only after training',
                      augmentation='same 224px canine pipeline as rounds4 and6',
                      concurrency='two jobs on physical GPU0 only; no throughput claims from training times'),
        replay=dict(reference=REFERENCE[('psan_ad0',42)], full_dataset_first_epoch=True,
                    criterion='train loss and validation Macro F1 agree after rounding to six decimal places'),
        statistics=dict(bootstrap=10000, seed=20261109,
                        ci='crossed visual-group by seed bootstrap, paired predictions and shared group resamples',
                        primary_family=['100-epoch AD0.2 minus AD0',
                                        '(100-epoch AD0.2 minus AD0) minus (30-epoch AD0.2 minus AD0)'],
                        correction='Holm across the two crossed contrasts',
                        per_seed='descriptive paired percentile intervals; no per-seed significance claims',
                        interpretation='post hoc within-corpus schedule/dropout evidence; no patient-level independence'),
        forbidden_selection_inputs=['test images', 'test labels', 'test scores'],
        source_code_sha256={p: sha256(ROOT / p) for p in SOURCE_FILES},
        environment=dict(torch=torch.__version__, cuda=torch.version.cuda))


def freeze():
    spec = protocol_spec()
    if PROTOCOL.exists() and json.loads(PROTOCOL.read_text()) != spec:
        raise ValueError('Frozen round7 dropout protocol/source changed; create a new protocol version')
    if not PROTOCOL.exists():
        write_json(PROTOCOL, spec)
    return spec


def load_protocol():
    spec = json.loads(PROTOCOL.read_text())
    if spec != protocol_spec():
        raise ValueError('Frozen round7 dropout protocol/source changed')
    return spec


@contextmanager
def training_context():
    """Only process-local protocol routing; the original training code stays unchanged."""
    old = training.PROTOCOL, training.protocol_spec
    training.PROTOCOL, training.protocol_spec = PROTOCOL, protocol_spec
    try:
        yield
    finally:
        training.PROTOCOL, training.protocol_spec = old


def gpu_guard():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
        raise ValueError('Set CUDA_VISIBLE_DEVICES=0; this experiment is restricted to physical GPU0')


def run(seed, *, output=OUT, limit=None, epochs=None):
    gpu_guard()
    load_protocol()
    with training_context():
        return training.train(job(seed), 'cuda:0', out=output, limit=limit, epochs=epochs)


def replay():
    directory = OUT / 'checks/replay42'
    result = run(42, output=directory, epochs=1)
    new = pd.read_csv(ROOT / Path(result['checkpoint']).parent / 'epochs.csv').iloc[0]
    old = pd.read_csv(ROOT / REFERENCE[('psan_ad0',42)] / 'epochs.csv').iloc[0]
    record = dict(reference=REFERENCE[('psan_ad0',42)], protocol_sha256=sha256(PROTOCOL),
                  train_loss_new=float(new.train_loss), train_loss_old=float(old.train_loss),
                  validation_f1_new=float(new.validation_macro_f1), validation_f1_old=float(old.validation_macro_f1),
                  agree_6dp=bool(round(new.train_loss,6)==round(old.train_loss,6) and
                                 round(new.validation_macro_f1,6)==round(old.validation_macro_f1,6)),
                  note='One epoch on the full original split; first warm-up LR matches the 100-epoch schedule.')
    write_json(OUT / 'checks/replay_first_epoch.json', record)
    print(json.dumps(record), flush=True)
    if not record['agree_6dp']:
        raise SystemExit('Full first-epoch replay differs; do not launch full runs')


def smoke():
    result = run(43, output=OUT / 'checks/smoke43', limit=256, epochs=1)
    record = dict(tag=result['tag'], epochs_completed=result['epochs_completed'],
                  parameters=result['parameters'], protocol_sha256=sha256(PROTOCOL),
                  test_used_for_selection=False, passed=result['epochs_completed']==1)
    write_json(OUT / 'checks/smoke.json', record)
    print(json.dumps(record), flush=True)


def launch():
    gpu_guard()
    load_protocol()
    checks = [OUT / 'checks/replay_first_epoch.json', OUT / 'checks/smoke.json']
    if not all(p.exists() for p in checks):
        raise ValueError('Replay and smoke checks must complete before launch')
    replay_record, smoke_record = [json.loads(p.read_text()) for p in checks]
    if not replay_record['agree_6dp'] or not smoke_record['passed']:
        raise ValueError('Replay/smoke check failed')
    if any(r['protocol_sha256'] != sha256(PROTOCOL) for r in (replay_record,smoke_record)):
        raise ValueError('Replay/smoke check belongs to another protocol')
    logs = OUT / 'logs'; logs.mkdir(parents=True,exist_ok=True)
    for seed in NEW_SEEDS:
        directory = OUT / 'long' / training.tag_of(job(seed))
        if (directory / 'summary.json').exists():
            print('Already complete',seed,flush=True)
            continue
        control = OUT / f'worker_{seed}.json'
        if control.exists():
            prior = json.loads(control.read_text())
            try:
                os.kill(prior['pid'],0)
                print('Already running',seed,prior['pid'],flush=True)
                continue
            except ProcessLookupError:
                pass
        log = logs / f'psan_ad0_seed{seed}.log'
        with log.open('a') as stream:
            proc = subprocess.Popen([sys.executable,'-u','-m','experiments.round7_dropout','run','--seed',str(seed)],
                                    cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,
                                    env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','OPENBLAS_NUM_THREADS':'4',
                                         'MKL_NUM_THREADS':'4','OMP_NUM_THREADS':'4'})
        write_json(control,dict(pid=proc.pid,seed=seed,log=str(log.relative_to(ROOT)),
                                resume_command=f'CUDA_VISIBLE_DEVICES=0 {sys.executable} -u -m '
                                               f'experiments.round7_dropout run --seed {seed}',
                                protocol_sha256=sha256(PROTOCOL)))
        print('Launched',seed,'pid',proc.pid,'log',log,flush=True)


def status():
    rows=[]
    for seed in NEW_SEEDS:
        directory=OUT/'long'/training.tag_of(job(seed))
        epochs=directory/'epochs.csv'
        row=dict(seed=seed,completed=(directory/'summary.json').exists())
        if epochs.exists():
            frame=pd.read_csv(epochs)
            row.update(epoch=int(frame.iloc[-1].epoch),best_validation_f1=float(frame.validation_macro_f1.max()),
                       mean_epoch_sec=float(frame.epoch_time_sec.mean()))
        rows.append(row)
    print(json.dumps(rows,indent=2),flush=True)


def analyze():
    """All declared contrasts are emitted regardless of their sign or significance."""
    from .round3_analysis import Evaluator,read_predictions,summary_predictions,tuned_runs,crossed,holm,strip
    spec=load_protocol()
    runs={key:json.loads((ROOT/path/'summary.json').read_text()) for key,path in REFERENCE.items()}
    for seed in NEW_SEEDS:
        runs[('psan_ad0',seed)]=json.loads((OUT/'long'/training.tag_of(job(seed))/'summary.json').read_text())
    for record in runs.values():
        if sha256(ROOT/record['checkpoint'])!=record['checkpoint_sha256']:
            raise ValueError('Changed analysis checkpoint')
        if record['epochs_completed']!=100 or record['split_sha256']!=spec['split_sha256']:
            raise ValueError('Incompatible long-training source')
    short=tuned_runs()
    ref=read_predictions(summary_predictions(runs[('psan',42)]))[0]
    evaluator=Evaluator(ref,b=spec['statistics']['bootstrap'],seed=spec['statistics']['seed'])
    rows=[]
    for seed in SEEDS:
        a,da=evaluator.score(summary_predictions(runs[('psan',seed)]),len(CLASSES))
        b,db=evaluator.score(summary_predictions(runs[('psan_ad0',seed)]),len(CLASSES))
        c,dc=evaluator.score(summary_predictions(short[('psan',seed)]),len(CLASSES))
        d,dd=evaluator.score(summary_predictions(short[('psan_ad0',seed)]),len(CLASSES))
        for name,delta,dist in [(spec['statistics']['primary_family'][0],a-b,da-db),
                                (spec['statistics']['primary_family'][1],(a-b)-(c-d),(da-db)-(dc-dd))]:
            rows.append(dict(comparison=name,seed=seed,delta_macro_f1=delta,
                             ci_low=float(np.quantile(dist,.025)),ci_high=float(np.quantile(dist,.975)),_dist=dist))
    contrasts=[crossed(rows,name,seed=spec['statistics']['seed']+1) for name in spec['statistics']['primary_family']]
    for record,pvalue in zip(contrasts,holm([r['p_crossed'] for r in contrasts])):
        record.update(p_holm=float(pvalue),family_size=2,post_hoc=True)
    tables=OUT/'tables'; tables.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(strip(rows)).to_csv(tables/'per_seed_contrasts.csv',index=False)
    pd.DataFrame(contrasts).to_csv(tables/'crossed_contrasts.csv',index=False)
    arm_rows=[dict(arm=a,seed=s,macro_f1=r['test']['macro_f1'],validation_macro_f1=r['best_validation_macro_f1'],
                   best_epoch=r['best_validation_epoch'],epochs_completed=r['epochs_completed'],
                   source=str(Path(r['checkpoint']).parent)) for (a,s),r in sorted(runs.items())]
    pd.DataFrame(arm_rows).to_csv(tables/'long_runs.csv',index=False)
    inputs={}
    for record in list(runs.values())+[short[(a,s)] for a in ['psan','psan_ad0'] for s in SEEDS]:
        p=summary_predictions(record); inputs[str(p.relative_to(ROOT))]=sha256(p)
    write_json(OUT/'analysis_provenance.json',dict(protocol_sha256=sha256(PROTOCOL),inputs=inputs,
                statistics=spec['statistics'],all_declared_contrasts_reported=True,
                note='Post hoc within-corpus comparison. Three seeds and unknown patient dependence limit inference.'))
    print(pd.DataFrame(contrasts).to_string(index=False),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['freeze','replay','smoke','launch','run','status','analyze'])
    parser.add_argument('--seed',type=int,choices=NEW_SEEDS)
    args=parser.parse_args()
    torch.set_num_threads(4)
    if args.command=='freeze':
        freeze();print('Frozen',sha256(PROTOCOL),flush=True)
    elif args.command=='run':
        if args.seed is None:parser.error('--seed is required for run')
        run(args.seed)
    else:
        globals()[args.command]()


if __name__=='__main__':
    main()
