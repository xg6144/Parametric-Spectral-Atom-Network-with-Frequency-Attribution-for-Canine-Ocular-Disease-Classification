"""Repetition of the frozen matched inference benchmark on an idle GPU without a display.

The round-7 session (experiments.round7_efficiency, protocol v3) ran on the card that drives the desktop
while other cards were training. This wrapper reuses its measurement code unchanged (same checkpoints,
inputs, warm-up, iterations, three fresh processes per cell) and changes only the session conditions:
physical GPU 0, no compute process on any card at session start, no display on the measured card.

    $PY -m experiments.round9_timing freeze
    $PY -m experiments.round9_timing run
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

from . import round7_efficiency as base
from .common import ROOT, sha256, write_json

OUT = ROOT / 'results/round9/timing'
PROTOCOL = ROOT / 'configs/round9_timing_protocol.json'
GPU_INDEX = '0'
_original_spec = base.protocol_spec


def protocol_spec():
    spec = _original_spec()
    card = next(r for r in base.gpu_inventory() if r['index'] == GPU_INDEX)
    spec.update(
        protocol='round9_matched_inference_idle_repetition_v1', frozen_on='2026-11-13',
        supersedes='none; repeats round7_matched_inference_v3, which remains reported',
        purpose='same matched-dense comparison under idle-host conditions, to separate run-to-run contention from model differences',
        gpu={k: card[k] for k in ['index', 'uuid', 'name', 'driver_version']},
        process_isolation='one fresh worker process per mode/batch/replicate; no compute process on any GPU at session start',
        host_caveat='no training on other GPUs during the session; the desktop display is driven by another card (GPU 1)',
        reused_measurement_source='experiments/round7_efficiency.py (cell and aggregate functions, unchanged)',
        wrapper_note='The reused cell function records physical_gpu_index=1 as a literal; the wrapper corrects this field to the measured card after each cell.')
    spec['source_code_sha256'] = dict(spec['source_code_sha256'],
                                      **{'experiments/round9_timing.py': sha256(ROOT / 'experiments/round9_timing.py')})
    return spec


# Point the reused module at this session's protocol and output directory.
base.OUT, base.PROTOCOL, base.protocol_spec = OUT, PROTOCOL, protocol_spec


def all_idle():
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'],
                            check=True, capture_output=True, text=True)
    return [row for row in result.stdout.strip().splitlines() if row.strip()]


def cell(mode, batch, replicate):
    base.cell(mode, batch, replicate)
    path = OUT / 'cells' / f'{mode}_b{batch}_r{replicate}.json'
    row = json.loads(path.read_text())
    row['physical_gpu_index'] = int(GPU_INDEX)
    write_json(path, row)


def run():
    spec = base.check_protocol()
    OUT.mkdir(parents=True, exist_ok=True)
    busy = all_idle()
    if busy:
        raise RuntimeError('Another GPU compute process is running; the idle repetition must start on an idle host: ' + repr(busy))
    write_json(OUT / 'session_start.json', {'gpu_inventory': base.gpu_inventory(), 'compute_processes': busy,
                                            'protocol_sha256': sha256(PROTOCOL)})
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = spec['gpu']['uuid']
    env['OMP_NUM_THREADS'] = env['MKL_NUM_THREADS'] = str(spec['cpu_threads'])
    for rep in spec['replicates']:
        order = base.MODES[rep - 1:] + base.MODES[:rep - 1]
        for batch in spec['batches']:
            for mode in order:
                path = OUT / 'cells' / f'{mode}_b{batch}_r{rep}.json'
                if path.exists():
                    if json.loads(path.read_text())['protocol_sha256'] != sha256(PROTOCOL):
                        raise ValueError('Cannot resume cell with another protocol')
                    continue
                subprocess.run([sys.executable, '-m', 'experiments.round9_timing', 'cell', '--mode', mode,
                                '--batch', str(batch), '--replicate', str(rep)], check=True, env=env)
    base.aggregate(spec)
    write_json(OUT / 'session_end.json', {'gpu_inventory': base.gpu_inventory(), 'compute_processes': all_idle()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['freeze', 'run', 'cell'])
    parser.add_argument('--mode', choices=base.MODES)
    parser.add_argument('--batch', type=int, choices=[1, 16, 64])
    parser.add_argument('--replicate', type=int, choices=[1, 2, 3])
    args = parser.parse_args()
    if args.action == 'freeze':
        base.freeze(); print('Frozen idle-host repetition protocol.', flush=True)
    elif args.action == 'run':
        run()
    else:
        cell(args.mode, args.batch, args.replicate)
