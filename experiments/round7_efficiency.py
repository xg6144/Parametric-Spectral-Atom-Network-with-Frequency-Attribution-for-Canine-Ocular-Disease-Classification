"""Reviewer-driven matched inference measurements, each in a fresh process.

    $PY -m experiments.round7_efficiency freeze
    $PY -m experiments.round7_efficiency run

Physical GPU 1 is reserved by coordination with the round-7 prior experiment.
The controller binds each worker to that GPU's UUID so workers cannot allocate
on the GPUs running other tasks. No model/training implementation is modified.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch

from .common import ROOT, build_model, load_model, seed_all, sha256, write_json


OUT = ROOT / 'results/round7/efficiency_v3'
PROTOCOL = ROOT / 'configs/round7_efficiency_v3_protocol.json'
MODES = ('psan_dynamic', 'psan_cached', 'dense')
RUNS = {
    'psan': ROOT / 'results/tuning/final/canine_psan_c2_s42_final',
    'dense': ROOT / 'results/round3/tuning/final/canine_gfnet_matched_c2_s42_final',
}
SOURCES = ('experiments/round7_efficiency.py', 'experiments/common.py',
           'models/psan.py', 'models/gfnet.py')


def gpu_inventory():
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=index,uuid,name,driver_version,memory.used,utilization.gpu',
         '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True)
    fields = ('index', 'uuid', 'name', 'driver_version', 'memory_used_mib', 'utilization_percent')
    return [dict(zip(fields, (x.strip() for x in row.split(','))))
            for row in result.stdout.strip().splitlines()]


def protocol_spec():
    cards = gpu_inventory()
    target = next(r for r in cards if r['index'] == '1')
    checkpoints = {}
    for arm, folder in RUNS.items():
        summary = json.loads((folder / 'summary.json').read_text())
        path = ROOT / summary['checkpoint']
        if sha256(path) != summary['checkpoint_sha256']:
            raise ValueError(f'Checkpoint hash differs from the archived selection: {arm}')
        checkpoints[arm] = {
            'checkpoint': str(path.relative_to(ROOT)), 'sha256': summary['checkpoint_sha256'],
            'summary_sha256': sha256(folder / 'summary.json'),
            'selection': 'earlier seed-42 validation-selected 30-epoch c2 EMA checkpoint',
        }
    historical = ROOT / 'results/selected_efficiency/cache_equivalence.json'
    equivalence = json.loads(historical.read_text())
    if equivalence['checkpoint_sha256'] != checkpoints['psan']['sha256']:
        raise ValueError('Historical full-test equality uses another PSAN checkpoint')
    return {
        'protocol': 'round7_matched_inference_v3', 'frozen_on': '2026-11-09',
        'supersedes': 'v1 source mismatch and v2 dense-checkpoint schema failure are excluded; v2 source/protocol/partial cells archived in revision_history/20261109_major_revision/efficiency_v2_interrupted.tar.gz',
        'loader_correction': 'Explicit matched-dense factory for its newer checkpoint config schema; strict archived EMA state loading, no model/training modification.',
        'reviewer_driven_post_hoc': True,
        'prior_exposure': ['submitted main and supplementary results, including earlier PSAN caching timings'],
        'purpose': 'direct matched-dense comparison; do not infer speed from filter parameter counts',
        'gpu': {k: target[k] for k in ('index', 'uuid', 'name', 'driver_version')},
        'checkpoints': checkpoints, 'modes': list(MODES), 'batches': [1, 16, 64],
        'replicates': [1, 2, 3], 'warmup': 20, 'iterations': 100,
        'precision': 'float32 tensors; autocast disabled; matmul TF32 false; cuDNN TF32 true',
        'input': 'identical seeded standard-normal tensors within each batch/replicate, RGB224x224',
        'input_seed': '42000 + 100*replicate + batch', 'cpu_threads': 4,
        'timing': 'host wall clock synchronized before/after each model-only call; output not retained',
        'memory': 'peak allocated and reserved bytes after warmup, including resident model/input/cache; unoccupied allocator slabs emptied before warmup',
        'memory_caveat': 'allocator measurements, not GPU process memory; atom parameters retained with cache',
        'cache_construction': 'cached cells:5 warmups then20 synchronized cache builds, separately timed',
        'equivalence': 'cached cells compare dynamic/cached synthetic logits at each batch/replicate; atol1e-5 and exact argmax',
        'historical_full_test_equality': {
            'source': str(historical.relative_to(ROOT)), 'sha256': sha256(historical),
            'images': equivalence['images'],
            'max_logit_abs_difference': equivalence['max_logit_abs_difference'],
            'argmax_agreement': equivalence['argmax_agreement'],
            'status': 'reused previously completed proof; not rerun in this timing experiment',
        },
        'cell_order': 'replicate blocks; mode order cyclically rotated by replicate, batch order1/16/64',
        'process_isolation': 'one fresh worker process per mode/batch/replicate; no concurrent compute process on GPU1',
        'host_caveat': 'other GPUs may run training and contend for CPU/host resources; desktop display active',
        'aggregation': 'median and range across the3 independent process medians; per-call median/q25/q75 retained per process',
        'excluded': ['checkpoint loading', 'I/O and preprocessing', 'cache construction from steady-state latency'],
        'source_code_sha256': {p: sha256(ROOT / p) for p in SOURCES},
    }


def freeze():
    spec = protocol_spec()
    if PROTOCOL.exists():
        if json.loads(PROTOCOL.read_text()) != spec:
            raise ValueError('Frozen efficiency protocol changed; preserve this run and create another version')
    else:
        write_json(PROTOCOL, spec)
    return spec


def check_protocol():
    spec = json.loads(PROTOCOL.read_text())
    if spec != protocol_spec():
        raise ValueError('Frozen efficiency sources/checkpoints differ')
    return spec


def no_other_compute(spec):
    result = subprocess.run(
        ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'],
        check=True, capture_output=True, text=True)
    for row in result.stdout.strip().splitlines():
        if not row.strip():
            continue
        uuid, pid = (x.strip() for x in row.split(','))
        if uuid == spec['gpu']['uuid'] and int(pid) != os.getpid():
            raise RuntimeError(f'GPU1 has another compute process ({pid}); retry after coordination')


def quantiles(values, prefix):
    return {f'{prefix}_{name}': float(np.percentile(values, q))
            for name, q in (('q25', 25), ('median', 50), ('q75', 75))}


@torch.no_grad()
def cell(mode, batch, replicate):
    spec = check_protocol()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != spec['gpu']['uuid']:
        raise RuntimeError('Workers must be launched by this controller on the reserved physical GPU1')
    no_other_compute(spec)
    torch.set_num_threads(spec['cpu_threads'])
    torch.cuda.set_device(0)
    seed_all(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    dev = torch.device('cuda:0')
    arm = 'dense' if mode == 'dense' else 'psan'
    checkpoint = spec['checkpoints'][arm]
    if arm == 'dense':
        saved = torch.load(ROOT / checkpoint['checkpoint'], map_location='cpu', weights_only=False)
        cfg = saved['config']
        model = build_model('gfnet_matched', classes=cfg['num_classes'], size=cfg['img_size'])
        state_key = 'eval_model'
        model.load_state_dict(saved[state_key], strict=True)
        model = model.to(dev).eval()
    else:
        model, saved, state_key = load_model(ROOT / checkpoint['checkpoint'], dev)
    del saved
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError('Expected float32 model parameters')
    generator = torch.Generator(device=dev).manual_seed(42000 + 100 * replicate + batch)
    x = torch.randn(batch, 3, 224, 224, generator=generator, device=dev, dtype=torch.float32)
    construction, equivalence = None, None
    sync = lambda: torch.cuda.synchronize(dev)
    if mode == 'psan_cached':
        model.clear_filter_cache()
        dynamic = model(x).detach().cpu()
        model.cache_filters(checkpoint['sha256'])
        cached = model(x).detach().cpu()
        maximum = float(torch.max(torch.abs(dynamic - cached)))
        agreement = bool(torch.equal(dynamic.argmax(1), cached.argmax(1)))
        if maximum > 1e-5 or not agreement:
            raise ValueError('Synthetic dynamic/cached equality failed')
        equivalence = {'max_logit_abs_difference': maximum, 'argmax_equal': agreement,
                       'images': batch, 'input_seed': 42000 + 100 * replicate + batch}
        del dynamic, cached
        for _ in range(5):
            model.cache_filters(checkpoint['sha256'])
        sync()
        construction_ms = []
        for _ in range(20):
            sync(); start = time.perf_counter()
            model.cache_filters(checkpoint['sha256'])
            sync(); construction_ms.append((time.perf_counter() - start) * 1000)
        construction = quantiles(construction_ms, 'cache_build_ms') | {'calls': 20, 'values_ms': construction_ms}
    # Remove unoccupied slabs left by cache construction/equivalence checks
    # before warmup; live model/input/cache tensors remain resident.
    torch.cuda.empty_cache()
    for _ in range(spec['warmup']):
        model(x)
    sync()
    torch.cuda.reset_peak_memory_stats(dev)
    allocated_before = torch.cuda.memory_allocated(dev)
    reserved_before = torch.cuda.memory_reserved(dev)
    times = []
    for _ in range(spec['iterations']):
        sync(); start = time.perf_counter(); model(x); sync()
        times.append((time.perf_counter() - start) * 1000)
    allocated_peak = torch.cuda.max_memory_allocated(dev)
    reserved_peak = torch.cuda.max_memory_reserved(dev)
    caches = [(name, value) for name, value in model.named_buffers()
              if name.endswith('_cached_filter') and value is not None]
    cache_bytes = sum(v.numel() * v.element_size() for _, v in caches)
    row = {
        'mode': mode, 'batch': batch, 'replicate': replicate,
        'checkpoint': checkpoint['checkpoint'], 'checkpoint_sha256': checkpoint['sha256'],
        'eval_state_key': state_key, 'precision': 'float32',
        'physical_gpu_index': 1, 'gpu_uuid': spec['gpu']['uuid'],
        'process_id': os.getpid(), 'pytorch': torch.__version__, 'cuda': torch.version.cuda,
        'cudnn': torch.backends.cudnn.version(), 'cpu_threads': spec['cpu_threads'],
        'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
        'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
        'parameters': sum(p.numel() for p in model.parameters()),
        'parameter_bytes': sum(p.numel() * p.element_size() for p in model.parameters()),
        'buffer_bytes': sum(v.numel() * v.element_size() for v in model.buffers()),
        'cache_bytes': cache_bytes, 'cache_tensors': len(caches),
        'allocated_before_bytes': allocated_before, 'reserved_before_bytes': reserved_before,
        'peak_allocated_bytes': allocated_peak, 'peak_reserved_bytes': reserved_peak,
        'warmup': spec['warmup'], 'iterations': spec['iterations'],
        **quantiles(times, 'latency_ms'),
        'latency_ms_mean': float(np.mean(times)),
        'throughput_images_per_sec_from_mean': float(batch * 1000 / np.mean(times)),
        'cache_construction': construction, 'synthetic_equivalence': equivalence,
        'latency_ms_values': times, 'gpu_inventory_after': gpu_inventory(),
        'protocol_sha256': sha256(PROTOCOL), 'source_code_sha256': sha256(__file__),
    }
    if (mode == 'psan_cached') != bool(cache_bytes):
        raise ValueError('Cache residency does not match the assigned condition')
    write_json(OUT / 'cells' / f'{mode}_b{batch}_r{replicate}.json', row)
    print(f'{mode} b{batch} r{replicate}: {row["latency_ms_median"]:.4f} ms; '
          f'peak={allocated_peak / 2**20:.2f} MiB', flush=True)


def aggregate(spec):
    rows = []
    for mode in MODES:
        for batch in spec['batches']:
            cells = [json.loads((OUT / 'cells' / f'{mode}_b{batch}_r{rep}.json').read_text())
                     for rep in spec['replicates']]
            if any(r['protocol_sha256'] != sha256(PROTOCOL) for r in cells):
                raise ValueError('Cell protocol mismatch')
            medians = [r['latency_ms_median'] for r in cells]
            row = {'mode': mode, 'batch': batch, 'replicates': len(cells),
                   'latency_ms_median_of_process_medians': float(np.median(medians)),
                   'latency_ms_min_process_median': min(medians),
                   'latency_ms_max_process_median': max(medians),
                   **quantiles(medians, 'process_median_ms'),
                   'median_within_process_iqr_ms': float(np.median([r['latency_ms_q75'] - r['latency_ms_q25'] for r in cells])),
                   'peak_allocated_mib_median': float(np.median([r['peak_allocated_bytes'] / 2**20 for r in cells])),
                   'peak_reserved_mib_median': float(np.median([r['peak_reserved_bytes'] / 2**20 for r in cells])),
                   'parameters': cells[0]['parameters'], 'parameter_bytes': cells[0]['parameter_bytes'],
                   'cache_bytes': cells[0]['cache_bytes'],
                   'cache_build_ms_median': float(np.median([r['cache_construction']['cache_build_ms_median'] for r in cells])) if mode == 'psan_cached' else None}
            rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / 'summary.csv', index=False)
    comparisons = []
    frame = pd.DataFrame(rows).set_index(['mode', 'batch'])
    for batch in spec['batches']:
        dense = frame.loc[('dense', batch)]
        for mode in MODES[:2]:
            psan = frame.loc[(mode, batch)]
            comparisons.append({'mode': mode, 'batch': batch,
                'latency_ratio_to_dense': float(psan.latency_ms_median_of_process_medians / dense.latency_ms_median_of_process_medians),
                'peak_allocated_difference_to_dense_mib': float(psan.peak_allocated_mib_median - dense.peak_allocated_mib_median)})
    pd.DataFrame(comparisons).to_csv(OUT / 'matched_comparisons.csv', index=False)
    write_json(OUT / 'provenance.json', {
        'status': 'completed', 'cells': 27, 'protocol_sha256': sha256(PROTOCOL),
        'source_code_sha256': sha256(__file__),
        'historical_full_test_equality': spec['historical_full_test_equality'],
        'synthetic_equality_cells': 9,
        'max_synthetic_logit_difference': max(json.loads(p.read_text())['synthetic_equivalence']['max_logit_abs_difference']
                                             for p in (OUT / 'cells').glob('psan_cached_*.json')),
        'summary_sha256': sha256(OUT / 'summary.csv'),
        'comparisons_sha256': sha256(OUT / 'matched_comparisons.csv'),
        'limitations': [spec['host_caveat'], spec['memory_caveat'],
                       'three process repeats quantify this session, not independent hardware populations',
                       'cache-build time reported separately; checkpoint loading and preprocessing excluded'],
    })
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


def run():
    spec = check_protocol()
    OUT.mkdir(parents=True, exist_ok=True)
    no_other_compute(spec)
    write_json(OUT / 'session_start.json', {'gpu_inventory': gpu_inventory(), 'protocol_sha256': sha256(PROTOCOL)})
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = spec['gpu']['uuid']
    env['OMP_NUM_THREADS'] = str(spec['cpu_threads'])
    env['MKL_NUM_THREADS'] = str(spec['cpu_threads'])
    for rep in spec['replicates']:
        shift = rep - 1
        order = MODES[shift:] + MODES[:shift]
        for batch in spec['batches']:
            for mode in order:
                path = OUT / 'cells' / f'{mode}_b{batch}_r{rep}.json'
                if path.exists():
                    if json.loads(path.read_text())['protocol_sha256'] != sha256(PROTOCOL):
                        raise ValueError('Cannot resume cell with another protocol')
                    continue
                subprocess.run([sys.executable, '-m', 'experiments.round7_efficiency', 'cell',
                                '--mode', mode, '--batch', str(batch), '--replicate', str(rep)],
                               check=True, env=env)
    aggregate(spec)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['freeze', 'run', 'cell', 'aggregate'])
    parser.add_argument('--mode', choices=MODES)
    parser.add_argument('--batch', type=int, choices=[1, 16, 64])
    parser.add_argument('--replicate', type=int, choices=[1, 2, 3])
    args = parser.parse_args()
    if args.action == 'freeze':
        freeze(); print('Frozen reviewer-driven matched inference protocol.', flush=True)
    elif args.action == 'run':
        run()
    elif args.action == 'aggregate':
        aggregate(check_protocol())
    else:
        if None in (args.mode, args.batch, args.replicate):
            parser.error('cell requires --mode, --batch and --replicate')
        cell(args.mode, args.batch, args.replicate)
