"""Post hoc check of the round-6 disease-only comparison (2026-10-15), specified after its test result was known.

In the frozen round-6 disease-only comparison (ten disease classes, every Normal image removed), PSAN was less accurate
than the matched dense control and the free rank-16 control. That comparison differs from the 11-class runs in two ways
that this check removes: (i) PSAN selected c1 on validation data while both controls selected c2, and (ii) with 25,916
instead of 62,422 training images, 30 epochs give only 42% of the optimizer steps of the 11-class runs, in which PSAN
matched the dense control at the corresponding point (epoch 12) and moved ahead only later. Here all three arms use c2
and are trained for 72 epochs (72 x 203 = 14,616 steps versus 30 x 488 = 14,640) with a 7-epoch warm-up (1,421 versus
1,464 steps), cosine decay over the full budget and no early stopping; the checkpoint is chosen by validation Macro F1
as in all rounds. Seeds 42-44. Everything else, including the training loop, is experiments.round6.train unchanged:
this module only wraps round6.run_config and writes to results/round6_posthoc/. This is a disclosed post hoc analysis.

    $PY -m experiments.round6_posthoc freeze
    $PY -m experiments.round6_posthoc worker --device cuda:0
    $PY -m experiments.round6_posthoc status
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time

import pandas as pd
import torch

from . import round6 as r6
from .common import ROOT, sha256, write_json

OUT = ROOT / 'results/round6_posthoc'
PROTOCOL = ROOT / 'configs/round6_posthoc_protocol.json'
ARMS = ['psan', 'gfnet_matched', 'lowrank16']
SEEDS = [42, 43, 44]
EPOCHS, WARMUP, CANDIDATE = 72, 7, 'c2'
SOURCES = ['experiments/round6_posthoc.py'] + r6.SOURCES


def protocol_spec():
    frame = r6.disease_frame()
    full = r6.load_manifest()
    train, train_full = int((frame.split == 'train').sum()), int((full.split == 'train').sum())
    steps = lambda images: -(-images // 128)
    return dict(
        protocol='round6_posthoc_disease_step_matched_v1',
        frozen_on='2026-10-15',
        post_hoc=True,
        motivation=('The frozen round-6 disease-only comparison found PSAN below the dense and free low-rank controls. '
                    'PSAN had selected c1 and both controls c2, and the disease-only runs received 42% of the optimizer '
                    'steps of the 11-class runs, all of them peaking in their last epoch. This check trains all three '
                    'arms with c2 and with as many optimizer steps as the 11-class 30-epoch runs.'),
        prior_exposure=['all round-2 to round-6 test results, including the round-6 disease-only test results'],
        arms=ARMS, seeds=SEEDS, candidate=CANDIDATE, epochs=EPOCHS, warmup_epochs=WARMUP, early_stopping=False,
        steps=dict(disease_train_images=train, eleven_class_train_images=train_full,
                   disease_steps=EPOCHS * steps(train), eleven_class_steps=30 * steps(train_full),
                   disease_warmup_steps=WARMUP * steps(train), eleven_class_warmup_steps=3 * steps(train_full)),
        unchanged=('data (round-6 disease-only split and labels), sampler, augmentation, optimizer, EMA, gradient clipping, '
                   'checkpoint rule and training loop of experiments.round6.train'),
        statistics=dict(bootstrap=10000, ci='crossed group x seed bootstrap across seeds 42-44',
                        correction='Holm within the post hoc family',
                        family=['psan minus gfnet_matched', 'psan minus lowrank16']),
        concurrency='one job per GPU in addition to the round-6 workers',
        source_code_sha256={p: sha256(ROOT / p) for p in SOURCES})


def freeze():
    spec = protocol_spec()
    if PROTOCOL.exists():
        if json.loads(PROTOCOL.read_text()) != spec:
            raise ValueError('Frozen post hoc protocol or source differs; create a new protocol version')
    else:
        write_json(PROTOCOL, spec)
    return spec


def load_protocol():
    spec = json.loads(PROTOCOL.read_text())
    if spec != protocol_spec():
        raise ValueError('Post hoc source/protocol changed after freezing')
    return spec


def patch(spec):
    """Wrap round6.run_config: same configuration except budget, warm-up, early stopping and identifiers."""
    original = r6.run_config

    def run_config(round6_spec, job, candidate, classes, split_sha):
        cfg = original(round6_spec, job, candidate, classes, split_sha)
        cfg.update(protocol=spec['protocol'], protocol_sha256=sha256(PROTOCOL), epochs=EPOCHS, warmup_epochs=WARMUP,
                   patience=None, post_hoc=True)
        return cfg
    r6.run_config = run_config


def jobs():
    return [dict(exp='dis', kind='final', arm=a, candidate=CANDIDATE, seed=s) for a in ARMS for s in SEEDS]


def worker(device):
    logs = OUT / 'logs'; logs.mkdir(parents=True, exist_ok=True)
    while True:
        load_protocol()
        states = [(job, *r6.job_state(job, OUT)) for job in jobs()]
        nxt = next(((job, tag) for job, state, tag in states if state == 'pending' and r6.acquire(tag, OUT)), None)
        if nxt is None:
            if all(state in ('done', 'failed') for _, state, _ in states):
                print('All post hoc jobs finished on', device, flush=True)
                return
            time.sleep(60)
            continue
        job, tag = nxt
        print(time.strftime('%H:%M:%S'), device, 'START', tag, flush=True)
        with (logs / f'{tag}.log').open('a') as log:
            code = subprocess.run([sys.executable, '-m', 'experiments.round6_posthoc', 'job', '--spec', json.dumps(job),
                                   '--device', device], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
        if code != 0:
            write_json(OUT / 'failed' / f'{tag}.json', dict(job=job, device=device, returncode=code))
        (OUT / 'locks' / f'{tag}.lock').unlink(missing_ok=True)
        print(time.strftime('%H:%M:%S'), device, 'END', tag, 'code', code, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['freeze', 'worker', 'job', 'status'])
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--spec')
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.command == 'freeze':
        freeze(); print('Frozen', sha256(PROTOCOL))
    elif args.command == 'worker':
        worker(args.device)
    elif args.command == 'job':
        patch(load_protocol())
        r6.train(json.loads(args.spec), args.device, out=OUT)
    elif args.command == 'status':
        load_protocol()
        print(pd.Series([r6.job_state(j, OUT)[0] for j in jobs()]).value_counts().to_string())


if __name__ == '__main__':
    main()
