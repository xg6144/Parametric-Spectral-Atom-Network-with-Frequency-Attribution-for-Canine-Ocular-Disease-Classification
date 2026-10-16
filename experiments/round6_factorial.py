"""Post hoc completion of the 2x2 design behind the round-6 disease-only comparison (2026-10-16), specified after the
results of experiments.round6 and experiments.round6_posthoc were known.

The step-matched post hoc check (round6_posthoc) changed the setting of PSAN (c1 -> c2) and its training length (30 ->
72 epochs) together, so it could not tell which change reversed the frozen disease-only result. An external re-review
asked for the two missing cells. This module trains PSAN on the ten disease classes, seeds 42-44, in
  c2_30  c2 (5e-4 / 0.05), 30 epochs, 3-epoch warm-up, early-stopping patience 10: exactly the configuration of the
         frozen round-6 disease-only finals of both controls, i.e. the run PSAN would have received had it selected c2;
  c1_72  c1 (3e-4 / 0.001), 72 epochs, 7-epoch warm-up, no early stopping: the configuration of the post hoc
         step-matched runs with the setting that PSAN had selected on validation data.
The other two cells exist: c1_30 (results/round6/disease/final, frozen) and c2_72 (results/round6_posthoc, post hoc).
Everything else, including the training loop, is experiments.round6.train unchanged; this module only wraps
round6.run_config and writes to results/round6_factorial/<cell>/. This is a disclosed post hoc analysis.

    $PY -m experiments.round6_factorial freeze
    $PY -m experiments.round6_factorial worker --device cuda:0
    $PY -m experiments.round6_factorial status
    $PY -m experiments.round6_factorial analyze
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

OUT = ROOT / 'results/round6_factorial'
PROTOCOL = ROOT / 'configs/round6_factorial_protocol.json'
SEEDS = [42, 43, 44]
CELLS = {'c2_30': dict(candidate='c2', epochs=30, warmup_epochs=3, patience=10),
         'c1_72': dict(candidate='c1', epochs=72, warmup_epochs=7, patience=None)}
SOURCES = ['experiments/round6_factorial.py'] + r6.SOURCES


def protocol_spec():
    return dict(
        protocol='round6_factorial_disease_v1',
        frozen_on='2026-10-16',
        post_hoc=True,
        motivation=('The step-matched post hoc disease-only check changed the setting of PSAN (c1 to c2) and the training '
                    'length (30 to 72 epochs) together. These runs complete the 2 x 2 design of setting and length for '
                    'PSAN so that the two changes can be separated.'),
        prior_exposure=['all round-2 to round-6 test results, including the frozen disease-only results and the post hoc '
                        'step-matched disease-only results'],
        arm='psan', seeds=SEEDS, cells=CELLS,
        existing_cells=dict(c1_30='results/round6/disease/final (frozen round-6 protocol)',
                            c2_72='results/round6_posthoc/disease/final (post hoc protocol)'),
        controls=dict(c2_30='gfnet_matched and lowrank16 in results/round6/disease/final (frozen round-6 protocol)',
                      c2_72='gfnet_matched and lowrank16 in results/round6_posthoc/disease/final (post hoc protocol)'),
        unchanged=('data (round-6 disease-only split and labels), sampler, augmentation, optimizer, EMA, gradient clipping, '
                   'checkpoint rule and training loop of experiments.round6.train'),
        statistics=dict(
            bootstrap=10000, ci='crossed group x seed bootstrap across seeds 42-44',
            families=dict(
                psan_vs_controls=['psan c2_30 minus gfnet_matched c2_30', 'psan c2_30 minus lowrank16 c2_30',
                                  'psan c1_72 minus gfnet_matched c2_72', 'psan c1_72 minus lowrank16 c2_72'],
                psan_factorial=['psan c2_30 minus psan c1_30', 'psan c2_72 minus psan c1_72',
                                'psan c1_72 minus psan c1_30', 'psan c2_72 minus psan c2_30']),
            correction='Holm within each family'),
        concurrency='up to two jobs per GPU on four otherwise idle GPUs',
        source_code_sha256={p: sha256(ROOT / p) for p in SOURCES})


def freeze():
    spec = protocol_spec()
    if PROTOCOL.exists():
        if json.loads(PROTOCOL.read_text()) != spec:
            raise ValueError('Frozen factorial protocol or source differs; create a new protocol version')
    else:
        write_json(PROTOCOL, spec)
    return spec


def load_protocol():
    spec = json.loads(PROTOCOL.read_text())
    if spec != protocol_spec():
        raise ValueError('Factorial source/protocol changed after freezing')
    return spec


def patch(spec, cell):
    """Wrap round6.run_config: same configuration except the cell's budget and the identifiers."""
    original = r6.run_config
    settings = CELLS[cell]

    def run_config(round6_spec, job, candidate, classes, split_sha):
        cfg = original(round6_spec, job, candidate, classes, split_sha)
        cfg.update(protocol=spec['protocol'], protocol_sha256=sha256(PROTOCOL), epochs=settings['epochs'],
                   warmup_epochs=settings['warmup_epochs'], patience=settings['patience'], post_hoc=True,
                   factorial_cell=cell)
        return cfg
    r6.run_config = run_config


def jobs():
    return [dict(exp='dis', kind='final', arm='psan', candidate=c['candidate'], seed=s, cell=cell)
            for cell, c in sorted(CELLS.items(), key=lambda item: -item[1]['epochs']) for s in SEEDS]


def out_of(job):
    return OUT / job['cell']


def worker(device):
    logs = OUT / 'logs'; logs.mkdir(parents=True, exist_ok=True)
    while True:
        load_protocol()
        states = [(job, *r6.job_state(job, out_of(job))) for job in jobs()]
        nxt = next(((job, tag) for job, state, tag in states if state == 'pending' and r6.acquire(tag, out_of(job))), None)
        if nxt is None:
            if all(state in ('done', 'failed') for _, state, _ in states):
                print('All factorial jobs finished on', device, flush=True)
                return
            time.sleep(60)
            continue
        job, tag = nxt
        print(time.strftime('%H:%M:%S'), device, 'START', job['cell'], tag, flush=True)
        with (logs / f"{job['cell']}_{tag}.log").open('a') as log:
            code = subprocess.run([sys.executable, '-m', 'experiments.round6_factorial', 'job', '--spec', json.dumps(job),
                                   '--device', device], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
        if code != 0:
            write_json(out_of(job) / 'failed' / f'{tag}.json', dict(job=job, device=device, returncode=code))
        (out_of(job) / 'locks' / f'{tag}.lock').unlink(missing_ok=True)
        print(time.strftime('%H:%M:%S'), device, 'END', job['cell'], tag, 'code', code, flush=True)


# ----------------------------------------------------------------------------- analysis (saved predictions only)

def cell_runs():
    """Disease-only summaries keyed by ('<arm> <cell>', seed) for the four PSAN cells and both control cells."""
    from .round6_analysis import POSTHOC, finals, load
    runs = {}
    for (arm, seed), s in finals('disease').items():
        cell = f"{s['candidate']}_30"
        if arm == 'psan' and cell != 'c1_30' or arm != 'psan' and cell != 'c2_30':
            raise ValueError('Unexpected frozen disease-only run ' + s['tag'])
        runs[(f'{arm} {cell}', seed)] = s
    for path in POSTHOC.glob('*/summary.json'):
        s = load(path)
        runs[(f"{s['arm']} c2_72", s['seed'])] = s
    for cell in CELLS:
        for path in (OUT / cell / 'disease' / 'final').glob('*/summary.json'):
            s = load(path)
            if s['arm'] != 'psan' or s['candidate'] != CELLS[cell]['candidate'] or s['max_epochs'] != CELLS[cell]['epochs']:
                raise ValueError('Factorial run does not match its cell: ' + str(path))
            runs[(f'psan {cell}', s['seed'])] = s
    return runs


def analyze():
    from .round3_analysis import SEED, Evaluator, read_predictions, strip, summary_predictions
    from .round6 import DISEASES
    from .round6_analysis import run_rows, seed_family
    spec = load_protocol()
    runs = cell_runs()
    expected = {(f'{arm} {cell}', s) for s in SEEDS for arm, cell in
                [('psan', 'c1_30'), ('psan', 'c2_30'), ('psan', 'c1_72'), ('psan', 'c2_72'), ('gfnet_matched', 'c2_30'),
                 ('lowrank16', 'c2_30'), ('gfnet_matched', 'c2_72'), ('lowrank16', 'c2_72')]}
    if set(runs) != expected:
        raise SystemExit('Factorial runs incomplete: missing ' + str(sorted(expected - set(runs))))
    reference, _, _ = read_predictions(summary_predictions(runs[('psan c1_30', 42)]))
    evaluator = Evaluator(reference, seed=SEED + 120)
    per_seed, crossed = [], []
    for family, contrasts in spec['statistics']['families'].items():
        pairs = [tuple(c.split(' minus ')) for c in contrasts]
        rows, out = seed_family(runs, pairs, len(DISEASES), evaluator, f'post hoc disease-only factorial, {family}')
        for r in rows + out:
            r['family'] = family
        per_seed += rows; crossed += out
    tables = OUT / 'tables'; tables.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(strip(per_seed)).to_csv(tables / 'factorial_per_seed.csv', index=False)
    pd.DataFrame(strip(crossed)).to_csv(tables / 'factorial_crossed.csv', index=False)
    rows = run_rows(runs, 'dis_factorial')
    for r in rows:
        r['arm'], r['cell'] = r['arm'].split(' ')
        r['max_epochs'] = runs[(f"{r['arm']} {r['cell']}", r['seed'])]['max_epochs']
    pd.DataFrame(rows).to_csv(tables / 'factorial_runs.csv', index=False)
    write_json(tables / 'factorial_provenance.json', dict(
        protocol_sha256=sha256(PROTOCOL), evaluator_seed=SEED + 120, runs={f'{k[0]}/{k[1]}': v['tag'] for k, v in sorted(runs.items())},
        source_sha256={p: sha256(ROOT / p) for p in ['experiments/round6_factorial.py', 'experiments/round6_analysis.py',
                                                     'experiments/round3_analysis.py']}))
    print('Factorial tables written to', tables)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['freeze', 'worker', 'job', 'status', 'analyze'])
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--spec')
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.command == 'freeze':
        freeze(); print('Frozen', sha256(PROTOCOL))
    elif args.command == 'worker':
        worker(args.device)
    elif args.command == 'job':
        job = json.loads(args.spec)
        patch(load_protocol(), job['cell'])
        r6.train(job, args.device, out=out_of(job))
    elif args.command == 'status':
        load_protocol()
        print(pd.Series([f"{j['cell']} {r6.job_state(j, out_of(j))[0]}" for j in jobs()]).value_counts().to_string())
    elif args.command == 'analyze':
        analyze()


if __name__ == '__main__':
    main()
