"""Post-hoc controlled intervention on unique training support, not compute.

BloodMNIST128 and corrected lesion-disjoint DermaMNIST128; fractions10/25/100%,
PSAN AD0 / matched dense / free rank16; seeds42--44. Each epoch samples the
same full-dataset number of examples with replacement, preserving expected
class exposure. Every cell uses the common c2 setting for30epochs.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from .common import ROOT, build_model, infer, prediction_csv, seed_all, sha256, write_json
from .round3 import external_data
from .round6 import NO_DECAY
from .training import ModelEMA, build_param_groups, seed_worker, train_one_epoch
from evaluation_metrics import compute_metric_bundle
from models.lowrank import to_low_rank

OUT = ROOT / 'results/round7/support'
PROTOCOL = ROOT / 'configs/round7_support_protocol.json'
DATASETS = ['bloodmnist128', 'dermamnist_c128']
SEEDS = [42, 43, 44]
FRACTIONS = [10, 25, 100]
ARMS = ['psan_ad0', 'gfnet_matched', 'lowrank16']
SOURCES = ['experiments/round7_support.py', 'experiments/round7_support_analysis.py', 'experiments/common.py', 'experiments/round3.py', 'experiments/round3_analysis.py',
           'experiments/round6.py', 'experiments/training.py', 'models/psan.py', 'models/gfnet.py',
           'models/lowrank.py', 'train/train_base.py', 'evaluation_metrics.py']


def choose_support(frame, fraction, seed):
    if not frame.groupby('group_id', observed=True).label.nunique().eq(1).all():
        raise ValueError('A training group has inconsistent labels.')
    groups = frame[['group_id', 'label']].drop_duplicates().sort_values(['label', 'group_id'])
    selected = []
    rng = np.random.default_rng(seed + 7000)
    for _, group in groups.groupby('label', sort=True):
        ordered = rng.permutation(group.group_id.to_numpy())
        count = max(1, math.ceil(len(ordered) * fraction / 100))
        selected.extend(ordered[:count].tolist())
    ids = np.flatnonzero(frame.group_id.isin(selected).to_numpy())
    return ids


def support_manifest(dataset, seed, fraction):
    frame = training_frame(dataset)
    ids = choose_support(frame, fraction, seed)
    result = frame.iloc[ids].copy()
    result.insert(0, 'training_index', ids)
    return result, len(frame)


@lru_cache(maxsize=2)
def training_frame(dataset):
    base = ROOT / 'results/external_data/round3_128'
    provenance = json.loads((base / f'{dataset}_provenance.json').read_text())
    path = base / (provenance['prepared_file'] if dataset == 'dermamnist_c128' else 'bloodmnist_128.npz')
    with np.load(path) as data:
        labels = data['train_labels'].reshape(-1)
        paths = [f'{dataset}/train/{i}' for i in range(len(labels))]
        groups = data['train_groups'] if 'train_groups' in data.files else paths
        return pd.DataFrame(dict(path=paths, label=labels, group_id=groups))


def protocol_spec():
    data = {}
    for dataset in DATASETS:
        prov = ROOT / 'results/external_data/round3_128' / f'{dataset}_provenance.json'
        data[dataset] = dict(provenance_sha256=sha256(prov),
                             grouping='lesion_id' if dataset == 'dermamnist_c128' else 'individual image',
                             supports={})
        for seed in SEEDS:
            for fraction in FRACTIONS:
                frame, full_n = support_manifest(dataset, seed, fraction)
                digest = hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest()
                data[dataset]['supports'][f's{seed}/f{fraction}'] = dict(
                    sha256=digest, images=len(frame), full_training_images=full_n,
                    unique_groups=int(frame.group_id.nunique()),
                    images_per_class={str(k): int(v) for k, v in frame.groupby('label').size().items()})
    return dict(
        protocol='round7_controlled_training_support_v1', frozen_on='2026-11-09', post_hoc=True,
        not_a_blinded_holdout=True,
        prior_exposure='All submitted canine/MedMNIST+ test results and reviewer comments were known. Reviewer-driven validation spectral diagnostics had already been computed; no support-intervention results existed.',
        question='Does the within-domain PSAN-control contrast change when unique training support is reduced at fixed optimization exposure?',
        datasets=DATASETS, seeds=SEEDS, nominal_support_percentages=FRACTIONS, arms=ARMS,
        supports='Within each class, shuffle whole training groups using seed+7000 and retain ceil(fraction*Nclassgroups); nested across fractions and common across arms.',
        sampler='Replacement sampler: full-training N draws per epoch in every support condition. Per-image weight sqrt(full_class_count)/subset_class_count preserves full-data expected class probabilities.',
        schedules=dict(epochs=30, early_stopping=False, warmup=3, cosine_over_full_budget=True,
                       lr=5e-4, weight_decay=.05, batch_size=128, num_workers=4, ema_decay=.999,
                       gradient_clip=1., stochastic_depth=.1, atom_dropout=0., atoms=16,
                       no_weight_decay=list(NO_DECAY), precision='float32; CUDA matmul TF32 off'),
        validation='Full unchanged validation split; best EMA validation Macro F1 checkpoint. No candidate screening or test selection.',
        test='Full unchanged official/corrected test split, evaluated once after validation selection.',
        initialization='Reset identical seed before each model construction. PSAN and lowrank start from identical filters; nonfilter tensors match all three arms.',
        execution='Two isolated processes on physical GPU1; fixed deterministic job shards; concurrency is not an inference/training-speed claim.',
        inference_limits='Intervention changes distinct training groups while holding expected class exposure and optimizer steps fixed; not a general causal explanation of between-domain differences.',
        statistics=dict(bootstrap=10000, seed=20261109,
                        resampling='Crossed support/initialization-seed and test-group bootstrap; lesion groups for Derma, images for Blood. Seed variation includes both subset choice and optimization.',
                        families=dict(simple='PSAN AD0 minus each of2 controls at each of3 fractions, per dataset:6 Holm-adjusted contrasts.',
                                      interactions='Difference of PSAN-control contrast at10 or25% versus100%, per dataset:4 Holm-adjusted contrasts.'),
                        all_cells='Report all54 cells, realized image/group counts, learning curves and controls; no outcome-based extension or dropping.'),
        data=data, source_code_sha256={p: sha256(ROOT / p) for p in SOURCES})


def freeze():
    spec = protocol_spec()
    proof_path = OUT / 'checks/design_checks.json'
    smoke_path = OUT / 'checks/smoke_checks.json'
    for path in [proof_path, smoke_path]:
        proof = json.loads(path.read_text())
        if not proof['passed'] or proof['support_source_sha256'] != sha256(__file__):
            raise ValueError('Current source requires successful design and smoke checks before freeze.')
    if PROTOCOL.exists() and json.loads(PROTOCOL.read_text()) != spec:
        raise ValueError('Frozen support protocol differs; create a new version.')
    if not PROTOCOL.exists(): write_json(PROTOCOL, spec)
    return spec


def make_model(arm, classes):
    # Callers reset the seed; constructors consume one common random stream.
    model = build_model('gfnet_matched' if arm == 'gfnet_matched' else 'psan',
                        classes=classes, size=128, dropout=0., atoms=16)
    if arm == 'lowrank16': to_low_rank(model, 16)
    return model


def tag(dataset, arm, seed, fraction):
    return f'{dataset}_{arm}_f{fraction}_c2_s{seed}_final'


def train(dataset, arm, seed, fraction, device_name, smoke=False, spec=None):
    if spec is None:
        spec = protocol_spec() if smoke else json.loads(PROTOCOL.read_text())
    directory = OUT / ('smoke' if smoke else 'final') / tag(dataset, arm, seed, fraction)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'summary.json').exists():
        result = json.loads((directory / 'summary.json').read_text())
        if sha256(ROOT / result['checkpoint']) != result['checkpoint_sha256']:
            raise ValueError('Completed support checkpoint changed.')
        return result
    splits = ['train', 'val'] if smoke else ['train', 'val', 'test']
    frames, datasets, classes, split_sha, _ = external_data(dataset, splits)
    full_counts = np.bincount(frames['train'].label, minlength=len(classes))
    full_n = len(frames['train'])
    ids = choose_support(frames['train'], fraction, seed)
    support = frames['train'].iloc[ids].copy(); support.insert(0, 'training_index', ids)
    expected = spec['data'][dataset]['supports'][f's{seed}/f{fraction}']
    if hashlib.sha256(support.to_csv(index=False).encode()).hexdigest() != expected['sha256']:
        raise ValueError('Support manifest changed.')
    support.to_csv(directory / 'support_manifest.csv', index=False)
    datasets['train'] = Subset(datasets['train'], ids.tolist())
    labels = support.label.to_numpy()
    if smoke:
        for split in ['val']:
            take = np.concatenate([np.flatnonzero(frames[split].label.to_numpy() == c)[:8] for c in range(len(classes))])
            datasets[split] = Subset(datasets[split], take.tolist())
            frames[split] = frames[split].iloc[take].reset_index(drop=True)
    cfg = dict(protocol=spec['protocol'], protocol_sha256='smoke' if smoke else sha256(PROTOCOL),
               dataset=dataset, arm=arm, seed=seed, support_percentage=fraction, candidate='c2',
               model='lowrank' if arm == 'lowrank16' else 'gfnet_matched' if arm == 'gfnet_matched' else 'psan',
               img_size=128, atoms=16, atom_dropout=0., num_classes=len(classes), classes=classes,
               epochs=1 if smoke else 30, warmup_epochs=3, lr=5e-4, weight_decay=.05,
               batch_size=128, num_workers=2 if smoke else 4, ema_decay=.999, gradient_clip=1.,
               sampler_draws_per_epoch=min(full_n, 256) if smoke else full_n,
               full_training_images=full_n, support_images=len(ids), support_groups=int(support.group_id.nunique()),
               split_sha256=split_sha, support_sha256=expected['sha256'],
               source_code_sha256=spec['source_code_sha256'], test_used_for_selection=False, smoke=smoke)
    config_path = directory / 'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != cfg:
        raise ValueError('Support run configuration changed.')
    write_json(config_path, cfg)
    fingerprint = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    seed_all(seed)
    device = torch.device(device_name)
    if device.type == 'cuda': torch.cuda.set_device(device)
    model = make_model(arm, len(classes)).to(device)
    optimizer = torch.optim.AdamW(build_param_groups(model, .05, NO_DECAY), lr=5e-4)
    ema = ModelEMA(model, .999, device)
    subset_counts = np.bincount(labels, minlength=len(classes))
    weights = np.sqrt(full_counts)[labels] / subset_counts[labels]
    sampler_rng = torch.Generator().manual_seed(seed + 1000)
    loader_rng = torch.Generator().manual_seed(seed + 2000)
    sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), cfg['sampler_draws_per_epoch'],
                                    replacement=True, generator=sampler_rng)
    loaders = {s: DataLoader(d, batch_size=128, sampler=sampler if s == 'train' else None,
                             shuffle=False, num_workers=cfg['num_workers'], pin_memory=device.type == 'cuda',
                             worker_init_fn=seed_worker, generator=loader_rng, persistent_workers=False)
               for s, d in datasets.items()}
    last_path, best_path = directory / 'last.pth', directory / 'best.pth'
    start, best, best_epoch, total, rows = 1, -1., 0, 0., []
    if last_path.exists():
        previous = torch.load(last_path, map_location=device, weights_only=False)
        if previous['config_fingerprint'] != fingerprint: raise ValueError('Resume fingerprint changed.')
        model.load_state_dict(previous['model']); ema.ema.load_state_dict(previous['ema'])
        optimizer.load_state_dict(previous['optimizer'])
        start, best, best_epoch = previous['epoch'] + 1, previous['best_macro_f1'], previous['best_epoch']
        total, rows, rng = previous['training_time_sec'], previous['history'], previous['rng']
        torch.set_rng_state(rng['torch'].cpu()); np.random.set_state(rng['numpy']); random.setstate(rng['python'])
        sampler_rng.set_state(rng['sampler'].cpu()); loader_rng.set_state(rng['loader'].cpu())
        if device.type == 'cuda': torch.cuda.set_rng_state(rng['cuda'].cpu(), device)
    for epoch in range(start, cfg['epochs'] + 1):
        tic = time.perf_counter()
        factor = epoch / 3 if epoch <= 3 else .5 * (1 + math.cos(math.pi * (epoch - 3) / 27))
        for group in optimizer.param_groups: group['lr'] = cfg['lr'] * factor
        loss, accuracy, extra = train_one_epoch(model, loaders['train'], optimizer, device, ema)
        logits, y = infer(ema.ema, loaders['val'], device)
        score = float(f1_score(y, logits.argmax(1), labels=range(len(classes)), average='macro', zero_division=0))
        if device.type == 'cuda': torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - tic; total += elapsed
        improved = score > best
        if improved: best, best_epoch = score, epoch
        rows.append(dict(epoch=epoch, train_loss=loss, train_accuracy=accuracy / 100,
                         validation_macro_f1=score, learning_rate=cfg['lr'] * factor,
                         epoch_time_sec=elapsed, training_time_sec=total,
                         gradient_norm_mean=extra['gradient_norm_mean']))
        checkpoint = dict(epoch=epoch, model=model.state_dict(), ema=ema.ema.state_dict(), eval_model=ema.ema.state_dict(),
                          optimizer=optimizer.state_dict(), config=cfg, config_fingerprint=fingerprint,
                          best_macro_f1=best, best_epoch=best_epoch, history=rows, training_time_sec=total,
                          rng=dict(torch=torch.get_rng_state(), numpy=np.random.get_state(), python=random.getstate(),
                                   sampler=sampler_rng.get_state(), loader=loader_rng.get_state(),
                                   cuda=torch.cuda.get_rng_state(device) if device.type == 'cuda' else None))
        temporary = last_path.with_suffix('.tmp'); torch.save(checkpoint, temporary); temporary.replace(last_path)
        if improved:
            temporary = best_path.with_suffix('.tmp'); torch.save(checkpoint, temporary); temporary.replace(best_path)
        pd.DataFrame(rows).to_csv(directory / 'epochs.csv', index=False)
        print(f'{directory.name} epoch={epoch}/{cfg["epochs"]} val_F1={score:.5f} time={elapsed:.1f}s', flush=True)
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=False)['eval_model']); model.eval()
    metrics = {}
    for split in splits[1:]:
        logits, y = infer(model, loaders[split], device)
        prediction_csv(directory / f'{split}_predictions.csv', frames[split], logits, y, classes)
        bundle = compute_metric_bundle(y, torch.as_tensor(logits).softmax(-1).numpy(), classes)
        write_json(directory / f'{split}_metrics.json', bundle); metrics[split] = bundle['metrics']
    summary = dict(tag=directory.name, dataset=dataset, arm=arm, seed=seed, support_percentage=fraction,
                   best_validation_macro_f1=best, best_validation_epoch=best_epoch, epochs_completed=rows[-1]['epoch'],
                   full_training_images=full_n, support_images=len(ids), support_groups=int(support.group_id.nunique()),
                   sampler_draws_per_epoch=cfg['sampler_draws_per_epoch'], training_time_sec=total,
                   checkpoint=str(best_path.relative_to(ROOT)), checkpoint_sha256=sha256(best_path),
                   config_fingerprint=fingerprint, protocol_sha256=cfg['protocol_sha256'],
                   split_sha256=split_sha, support_sha256=expected['sha256'], smoke=smoke, **metrics)
    write_json(directory / 'summary.json', summary)
    last_path.unlink(missing_ok=True)
    del model, ema, optimizer, loaders, datasets
    gc.collect()
    if device.type == 'cuda': torch.cuda.empty_cache()
    return summary


def gpu_guard(device):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '1' or device != 'cuda:0':
        raise ValueError('Set CUDA_VISIBLE_DEVICES=1 and --device cuda:0; only physical GPU1 is authorized.')


def design_checks():
    records = []
    for dataset in DATASETS:
        frames, _, classes, _, _ = external_data(dataset, ['train', 'val', 'test'])
        for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
            if set(frames[a].group_id) & set(frames[b].group_id):
                raise ValueError('Training/validation/test groups overlap.')
        reference = training_frame(dataset)
        if not reference[['path', 'label', 'group_id']].equals(frames['train'][['path', 'label', 'group_id']]):
            raise ValueError('Cached support-metadata frame differs from actual loader.')
        full_counts = np.bincount(reference.label, minlength=len(classes))
        for seed in SEEDS:
            nested = [choose_support(reference, f, seed) for f in FRACTIONS]
            if not set(nested[0]) <= set(nested[1]) <= set(nested[2]):
                raise ValueError('Support groups are not nested.')
            for fraction, ids in zip(FRACTIONS, nested):
                chosen = reference.iloc[ids]
                subset_counts = np.bincount(chosen.label, minlength=len(classes))
                weights = np.sqrt(full_counts)[chosen.label] / subset_counts[chosen.label]
                class_mass = np.bincount(chosen.label, weights=weights, minlength=len(classes))
                np.testing.assert_allclose(class_mass, np.sqrt(full_counts), rtol=1e-12, atol=1e-12)
                if set(ids) != set(np.flatnonzero(reference.group_id.isin(chosen.group_id))):
                    raise ValueError('A partial training group was selected.')
                records.append(dict(dataset=dataset, seed=seed, fraction=fraction, images=len(ids),
                                    groups=int(chosen.group_id.nunique()), expected_class_masses_match=True))
        models = {}
        for arm in ARMS:
            seed_all(42); models[arm] = make_model(arm, len(classes)).eval()
        donor = {k: v for k, v in models['psan_ad0'].state_dict().items() if '.filter.' not in k}
        for arm in ['gfnet_matched', 'lowrank16']:
            other = {k: v for k, v in models[arm].state_dict().items() if '.filter.' not in k}
            if donor.keys() != other.keys() or not all(torch.equal(v, other[k]) for k, v in donor.items()):
                raise ValueError('Non-filter initialization differs across arms.')
        with torch.no_grad():
            seed_all(123); x = torch.randn(2, 3, 128, 128)
            a, b = models['psan_ad0'](x), models['lowrank16'](x)
            if not torch.allclose(a, b, atol=1e-5, rtol=1e-5):
                raise ValueError('PSAN and rank16 initial functions differ.')
        del models
    write_json(OUT / 'checks/design_checks.json', dict(passed=True, records=records,
               support_source_sha256=sha256(__file__), nonfilter_initialization='identical all3arms',
               initial_psan_lowrank='allclose atol/rtol1e-5', group_disjointness=True,
               support_intervention_results_read=False))
    print('DESIGN CHECKS PASSED', len(records), 'support cells', flush=True)


def smoke(device):
    gpu_guard(device)
    spec = protocol_spec(); records = []
    for dataset in DATASETS:
        for arm in ARMS:
            result = train(dataset, arm, 42, 10, device, smoke=True, spec=spec)
            path = ROOT / result['checkpoint']
            passed = result['epochs_completed'] == 1 and np.isfinite(result['best_validation_macro_f1'])
            if (path.parent / 'test_predictions.csv').exists():
                raise ValueError('Smoke must not inspect test predictions.')
            records.append(dict(tag=result['tag'], passed=bool(passed), epochs=1,
                                test_predictions_generated=False))
    write_json(OUT / 'checks/smoke_checks.json', dict(passed=all(r['passed'] for r in records),
               records=records, support_source_sha256=sha256(__file__)))
    print('SMOKE CHECKS PASSED', len(records), 'model/domain cells', flush=True)


def worker(device, shard=0, shards=2):
    gpu_guard(device)
    spec = json.loads(PROTOCOL.read_text())
    if spec != protocol_spec(): raise ValueError('Source or data changed after support protocol freeze.')
    jobs = [(d, a, s, f) for s in SEEDS for d in DATASETS for f in FRACTIONS for a in ARMS]
    for index, (dataset, arm, seed, fraction) in enumerate(jobs):
        if index % shards != shard: continue
        result = train(dataset, arm, seed, fraction, device, spec=spec)
        print('COMPLETED', result['tag'], 'test_F1', result['test']['macro_f1'], flush=True)


def pid_alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def progress():
    runs = sorted((OUT / 'final').glob('*/summary.json'))
    workers = []
    for path in sorted(OUT.glob('worker_*.json')):
        entry = json.loads(path.read_text())
        entry['alive'] = pid_alive(entry['pid'])
        workers.append(entry)
    return dict(completed=len(runs), planned=54, workers=workers,
                updated_kst=datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),
                runs=[p.parent.name for p in runs])


def launch():
    spec = json.loads(PROTOCOL.read_text())
    if spec != protocol_spec(): raise ValueError('Source or data changed after support protocol freeze.')
    OUT.mkdir(parents=True, exist_ok=True); (OUT / 'logs').mkdir(exist_ok=True)
    environment = os.environ.copy(); environment['CUDA_VISIBLE_DEVICES'] = '1'
    environment['OPENBLAS_NUM_THREADS'] = '4'; environment['OMP_NUM_THREADS'] = '4'
    for shard in range(2):
        info_path = OUT / f'worker_{shard}.json'
        if info_path.exists() and pid_alive(json.loads(info_path.read_text())['pid']): continue
        command = [sys.executable, '-m', 'experiments.round7_support', 'worker', '--device', 'cuda:0',
                   '--shard', str(shard), '--shards', '2']
        with (OUT / 'logs' / f'worker_{shard}.log').open('a') as log:
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        write_json(info_path, dict(pid=process.pid, shard=shard, command=command,
                                  physical_gpu=1, protocol_sha256=sha256(PROTOCOL)))
    monitor_path = OUT / 'monitor.json'
    if not monitor_path.exists() or not pid_alive(json.loads(monitor_path.read_text())['pid']):
        with (OUT / 'logs' / 'monitor.log').open('a') as log:
            process = subprocess.Popen([sys.executable, '-m', 'experiments.round7_support', 'monitor'],
                cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        write_json(monitor_path, dict(pid=process.pid, protocol_sha256=sha256(PROTOCOL)))
    print(json.dumps(progress(), indent=2), flush=True)


def monitor():
    while True:
        state = progress()
        state['state'] = 'training'
        if state['completed'] == 54:
            state['state'] = 'analyzing'; write_json(OUT / 'completion_status.json', state)
            with (OUT / 'logs' / 'analysis.log').open('a') as log:
                result = subprocess.run([sys.executable, '-m', 'experiments.round7_support_analysis'],
                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            state['state'] = 'complete' if result.returncode == 0 else 'analysis_failed'
            state['analysis_exit_code'] = result.returncode
            write_json(OUT / 'completion_status.json', state)
            return
        if state['workers'] and not any(w['alive'] for w in state['workers']):
            state['state'] = 'training_failed'; write_json(OUT / 'completion_status.json', state)
            raise RuntimeError('All support workers stopped before all54cells completed; inspect logs and resume.')
        write_json(OUT / 'completion_status.json', state)
        time.sleep(30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['checks', 'freeze', 'worker', 'smoke', 'status', 'launch', 'monitor'])
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.command == 'freeze': freeze(); print(PROTOCOL)
    elif args.command == 'checks': design_checks()
    elif args.command == 'worker': worker(args.device, args.shard, args.shards)
    elif args.command == 'launch': launch()
    elif args.command == 'monitor': monitor()
    elif args.command == 'status':
        print(json.dumps(progress(), indent=2))
    else:
        smoke(args.device)


if __name__ == '__main__': main()
