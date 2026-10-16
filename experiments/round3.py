"""Round-3 follow-up (2026-10-14): controlled dense-mixer tuning, selected-setting seed
replication, OFAT atom-count replication and MedMNIST+128 architecture benchmarks.

Outputs are written under results/round3/ so the frozen round-2 aggregation is unchanged.
The epoch loop mirrors experiments/validation_tuning.train_trial and
experiments/training.train_one: the same data pipeline, inverse-square-root class sampler,
AdamW parameter groups, warmup+cosine schedule, gradient clipping, EMA validation,
validation-Macro-F1 checkpoint selection and final evaluation of the selected EMA weights.
Screening never constructs a test dataset. Earlier canine test outcomes were observed before
this protocol was frozen, so it is a disclosed follow-up, not a blinded holdout.

    $PY -m experiments.round3 freeze
    $PY -m experiments.round3 prepare            # verify/prepare MedMNIST+128 files
    $PY -m experiments.round3 worker --device cuda:0
    $PY -m experiments.round3 status
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, WeightedRandomSampler

from .common import (CLASSES, ROOT, ManifestDataset, build_model, infer, load_manifest,
                     prediction_csv, seed_all, sha256, write_json)
from .training import ExternalDataset, ModelEMA, build_param_groups, seed_worker, train_one_epoch
from evaluation_metrics import compute_metric_bundle

OUT = ROOT / 'results/round3'
PROTOCOL = ROOT / 'configs/round3_protocol.json'
EXTERNAL_ROOT = ROOT / 'results/external_data/round3_128'
SOURCES = ['experiments/round3.py', 'experiments/common.py', 'experiments/training.py',
           'models/psan.py', 'models/gfnet.py', 'models/afno.py', 'train/train_base.py',
           'evaluation_metrics.py']
NO_DECAY = ('norm', 'filter_real', 'filter_imag', 'mu', 'log_sigma', 'rho_raw',
            'theta', 'log_amp', 'phi', 'w_real', 'w_imag')
CANDIDATES = [dict(id='c0', lr=1e-4, weight_decay=1e-3),
              dict(id='c1', lr=3e-4, weight_decay=1e-3),
              dict(id='c2', lr=5e-4, weight_decay=5e-2)]
ARMS = {'psan': dict(name='psan', atom_dropout=.2, atoms=16),
        'psan_ad0': dict(name='psan', atom_dropout=0., atoms=16),
        'gfnet_matched': dict(name='gfnet_matched'),
        'gfnet': dict(name='gfnet'),
        'afno': dict(name='afno')}
EXTERNAL = {'dermamnist_c128': dict(source='dermamnist', md5='2defd784463fa5243564e855ed717de1'),
            'bloodmnist128': dict(source='bloodmnist', md5='adace1e0ed228fccda1f39692059dd4c'),
            'octmnist128': dict(source='octmnist', md5='0a97e76651ace45c5d943ee3f65b63ae')}
DERMA_LABELS = {'akiec': 0, 'bcc': 1, 'bkl': 2, 'df': 3, 'mel': 4, 'nv': 5, 'vasc': 6}
COMMON_RUNS = {'gfnet_matched': 'canine_gfnet_matched_s{}',
               'psan_ad0': 'canine_psan_gabor_ad0p0_M16_morlet_s{}'}


def protocol_spec():
    selections = {m: json.loads((ROOT / 'results/tuning/selections' / f'{m}.json').read_text())['candidate']['id']
                  for m in ['psan', 'gfnet', 'afno']}
    return dict(
        protocol='round3_controlled_dense_replication_external128_v1',
        frozen_on='2026-10-14',
        motivation=('The round-2 frozen PSAN uses atom dropout 0.2 while the matched dense mixer has no filter '
                    'dropout, and round-2 factorial runs showed lower Macro F1 with dropout under the 30-epoch '
                    'budget. This follow-up gives the matched dense control and a no-dropout PSAN arm the same '
                    'validation-only screening, replicates the round-2 selected settings over seeds, replicates '
                    'the M32/M64 OFAT axis, and repeats the external architecture benchmarks at 128 pixels, '
                    'where F=40 exceeds M=16.'),
        prior_exposure=[
            'canine common-preset test results of all round-2 runs, including PSAN AD0 versus matched dense (seeds 42-44)',
            'canine seed-42 validation-selected test results of the eight round-2 arms',
            'canine seed-42 OFAT M32/M64 test results',
            'MedMNIST+64 common-preset test results of PSAN/GFNet/AFNO'],
        not_a_blinded_holdout=True,
        canine=dict(
            split_sha256=sha256(ROOT / 'results/revision_split.csv'),
            screened_arms=['gfnet_matched', 'psan_ad0'], screening_seed=42,
            screened_final_seeds=[42, 43, 44],
            replicated_round2_selections=selections, replication_seeds=[43, 44],
            ofat_replication=dict(
                atoms=[32, 64], atom_dropout=0., seeds=[43, 44], candidate='c0 (common preset)',
                initialization='non-filter tensors copied from an untrained PSAN M16 built from the same seed '
                               '(round-2 OFAT donor rule)'),
            c0_reuse='a selected c0 final reuses the hash-verified identical common-preset run in results/training'),
        external=dict(
            datasets=list(EXTERNAL), arms=list(ARMS), img_size=128, patch=16, token_grid=[8, 8],
            frequency_bins=40, atoms=16, screening_seed=42, final_seeds=[42, 43, 44],
            dermamnist_c128=('corrected lesion-disjoint rows of results/external_data/dermamnist_c64_*_identity.csv '
                             'mapped to the official 128-pixel rasters by official split and index'),
            published_md5={k: v['md5'] for k, v in EXTERNAL.items()}),
        candidates=CANDIDATES, screening_epochs=8, final_epochs=30, final_patience=10,
        warmup_epochs=3, batch_size=128, num_workers=6, ema_decay=.999, gradient_clip=1.,
        precision='float32 tensors; CUDA matmul TF32 off; cuDNN convolution TF32 default',
        stochastic_depth=.1, sampler='inverse-square-root class-weighted replacement sampling',
        objective='highest best-epoch validation Macro F1; ties use earlier candidate order',
        screening_early_stopping=False,
        forbidden_selection_inputs=['test images', 'test labels', 'test predictions or metrics'],
        statistics=dict(
            bootstrap=10000,
            ci='paired percentile bootstrap over visual groups (canine), lesions (corrected Derma) or images',
            across_seeds='crossed bootstrap of evaluation groups and seed indices; seed-level paired t descriptive',
            correction='Holm within each declared family',
            families=dict(
                canine_tuned=['psan-gfnet_matched', 'psan_ad0-gfnet_matched', 'psan-gfnet', 'psan-afno'],
                canine_ofat_replication=['M32-M16', 'M64-M16'],
                external_each_dataset=['psan-gfnet_matched', 'psan_ad0-gfnet_matched', 'psan-gfnet', 'psan-afno'])),
        checkpoint_retention='best.pth kept; resume-only last.pth deleted after a completed run to bound disk use',
        concurrency='up to two jobs per GPU; wall-clock times are contended and not reported as cost',
        source_code_sha256={p: sha256(ROOT / p) for p in SOURCES})


def freeze():
    spec = protocol_spec()
    if PROTOCOL.exists():
        if json.loads(PROTOCOL.read_text()) != spec:
            raise ValueError('Frozen round-3 protocol or source differs; create a new protocol version')
    else:
        write_json(PROTOCOL, spec)
    return spec


def load_protocol():
    spec = json.loads(PROTOCOL.read_text())
    if spec != protocol_spec():
        raise ValueError('Round-3 source/protocol changed after freezing')
    return spec


def tag_of(job, candidate=None):
    cid = candidate or job['candidate']
    if job['kind'] == 'ofat':
        return f"canine_psan_gabor_ad0p0_M{job['atoms']}_morlet_s{job['seed']}"
    suffix = 'screening' if job['kind'] == 'screen' else 'final'
    return f"{job['dataset']}_{job['arm']}_{cid}_s{job['seed']}_{suffix}"


def run_dir(job, candidate=None, out=OUT):
    if job['kind'] == 'ofat':
        return out / 'training' / tag_of(job)
    family = 'tuning' if job['dataset'] == 'canine' else 'external'
    stage = 'trials' if job['kind'] == 'screen' else 'final'
    return out / family / stage / tag_of(job, candidate)


def selection_path(dataset, arm, out=OUT):
    return out / 'selections' / f'{dataset}_{arm}.json'


def resolved_candidate(job, out=OUT):
    if job['candidate'] is not None:
        return job['candidate']
    path = selection_path(job['dataset'], job['arm'], out)
    return json.loads(path.read_text())['candidate']['id'] if path.exists() else None


def job_list(spec):
    jobs = []
    canine, external = spec['canine'], spec['external']
    screens = [dict(kind='screen', dataset='canine', arm=a, candidate=c['id'], seed=42)
               for a in canine['screened_arms'] for c in CANDIDATES]
    replicated = [dict(kind='final', dataset='canine', arm=a, candidate=c, seed=s)
                  for a, c in canine['replicated_round2_selections'].items() for s in canine['replication_seeds']]
    jobs += screens + [j for j in replicated if j['arm'] == 'psan']
    jobs += [dict(kind='screen', dataset=d, arm=a, candidate=c['id'], seed=42)
             for d in external['datasets'] for a in external['arms'] for c in CANDIDATES]
    jobs += [dict(kind='final', dataset='canine', arm=a, candidate=None, seed=s)
             for a in canine['screened_arms'] for s in canine['screened_final_seeds']]
    jobs += [j for j in replicated if j['arm'] != 'psan']
    jobs += [dict(kind='final', dataset=d, arm=a, candidate=None, seed=s)
             for d in external['datasets'] for a in external['arms'] for s in external['final_seeds']]
    ofat = canine['ofat_replication']
    jobs += [dict(kind='ofat', dataset='canine', arm=f'psan_ad0_M{m}', candidate='c0', seed=s, atoms=m)
             for m in ofat['atoms'] for s in ofat['seeds']]
    return jobs


# ----------------------------------------------------------------------------- data

def prepare_external():
    """Verify official MedMNIST+128 files and build the corrected lesion-disjoint Derma split."""
    from medmnist.info import INFO
    for dataset, item in EXTERNAL.items():
        source = item['source']
        official = EXTERNAL_ROOT / f'{source}_128.npz'
        provenance_path = EXTERNAL_ROOT / f'{dataset}_provenance.json'
        if provenance_path.exists() or not official.exists():
            continue
        digest = hashlib.md5(official.read_bytes()).hexdigest()
        if digest != item['md5'] or digest != INFO[source]['MD5_128']:
            raise ValueError(f'Published MD5 mismatch for {official.name}')
        npz = np.load(official)
        arrays = {key: npz[key] for key in npz.files}
        info = INFO[source]
        provenance = dict(source=info['url_128'], published_md5=info['MD5_128'], actual_md5=digest,
                          official_sha256=sha256(official), license=info['license'],
                          description=info['description'], resolution=128,
                          labels=info['label'], clinical_canine_external_validation=False)
        if dataset == 'dermamnist_c128':
            low = np.load(ROOT / 'results/external_data/dermamnist_64.npz')
            low = {key: low[key] for key in low.files}
            prepared, sizes, correlations = {}, {}, []
            lesions = {}
            for split in ['train', 'val', 'test']:
                identity = pd.read_csv(ROOT / 'results/external_data' / f'dermamnist_c64_{split}_identity.csv')
                rows = list(identity.itertuples(index=False))
                images = np.stack([arrays[r.split_official + '_images'][r.index] for r in rows])
                labels = identity.dx.map(DERMA_LABELS).to_numpy()[:, None]
                official_labels = np.array([arrays[r.split_official + '_labels'][r.index, 0] for r in rows])
                if not np.array_equal(official_labels, labels[:, 0]):
                    raise ValueError('Corrected Derma labels disagree with official 128 labels')
                # The 128 and 64 releases must index the same lesion images.
                for r, image in zip(rows, images):
                    reduced = np.asarray(Image.fromarray(image).resize((64, 64), Image.BILINEAR), dtype=np.float64)
                    reference = low[r.split_official + '_images'][r.index].astype(np.float64)
                    correlations.append(np.corrcoef(reduced.ravel(), reference.ravel())[0, 1])
                prepared[split + '_images'] = images
                prepared[split + '_labels'] = labels
                prepared[split + '_groups'] = identity.lesion_id.to_numpy().astype(str)
                sizes[split] = len(identity)
                lesions[split] = set(identity.lesion_id)
            correlations = np.asarray(correlations)
            if correlations.min() < .9:
                raise ValueError(f'128/64 raster correspondence failed (min r={correlations.min():.3f})')
            overlap = {a + '_' + b: len(lesions[a] & lesions[b])
                       for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]}
            if any(overlap.values()):
                raise ValueError('Lesion overlap in corrected Derma split')
            target = EXTERNAL_ROOT / 'dermamnist_c128_prepared.npz'
            np.savez(target, **prepared)
            provenance.update(prepared_file=target.name, prepared_sha256=sha256(target), split_sizes=sizes,
                              lesion_overlap=overlap, label_mapping=DERMA_LABELS,
                              raster_correspondence_min_r=float(correlations.min()),
                              raster_correspondence_median_r=float(np.median(correlations)),
                              mapping='results/external_data/dermamnist_c64_{train,val,test}_identity.csv',
                              patient_independence_established=False, lesion_group_split_verified=True)
        else:
            hashes = {s: {hashlib.sha256(x.tobytes()).hexdigest() for x in arrays[s + '_images']}
                      for s in ['train', 'val', 'test']}
            provenance.update(split_sizes={s: int(len(arrays[s + '_labels'])) for s in ['train', 'val', 'test']},
                              pixel_hash_cross_split_overlap={a + '_' + b: len(hashes[a] & hashes[b]) for a, b in
                                                              [('train', 'val'), ('train', 'test'), ('val', 'test')]},
                              patient_or_lesion_ids_in_npz=False)
        write_json(provenance_path, provenance)
        print('Prepared', dataset, provenance.get('split_sizes'), flush=True)


def canine_data(splits, limit=None):
    frame = load_manifest()
    frames = {s: frame[frame.split == s].reset_index(drop=True) for s in splits}
    if limit:
        frames = {s: f.iloc[:limit].reset_index(drop=True) for s, f in frames.items()}
    datasets = {s: ManifestDataset(f, s == 'train', 224) for s, f in frames.items()}
    return frames, datasets, CLASSES, sha256(ROOT / 'results/revision_split.csv'), 224


def external_data(dataset, splits, limit=None):
    from medmnist.info import INFO
    provenance_path = EXTERNAL_ROOT / f'{dataset}_provenance.json'
    provenance = json.loads(provenance_path.read_text())
    source = EXTERNAL[dataset]['source']
    if dataset == 'dermamnist_c128':
        path = EXTERNAL_ROOT / provenance['prepared_file']
        if sha256(path) != provenance['prepared_sha256']:
            raise ValueError('Prepared Derma file changed')
    else:
        path = EXTERNAL_ROOT / f'{source}_128.npz'
        if sha256(path) != provenance['official_sha256']:
            raise ValueError('Official external file changed')
    npz = np.load(path)
    frames, datasets = {}, {}
    for split in splits:
        images, labels = npz[split + '_images'], npz[split + '_labels'].reshape(-1)
        if limit:
            images, labels = images[:limit], labels[:limit]
        paths = [f'{dataset}/{split}/{i}' for i in range(len(labels))]
        groups = npz[split + '_groups'][:len(labels)] if split + '_groups' in npz.files else paths
        frames[split] = pd.DataFrame({'path': paths, 'label': labels, 'group_id': groups})
        datasets[split] = ExternalDataset(images, labels, split == 'train', 128)
    classes = [INFO[source]['label'][str(i)] for i in range(len(INFO[source]['label']))]
    return frames, datasets, classes, sha256(provenance_path), 128


# ----------------------------------------------------------------------------- training

def run_config(spec, job, candidate, size, classes, split_sha):
    arm = ARMS['psan_ad0'] if job['kind'] == 'ofat' else ARMS[job['arm']]
    cand = next(c for c in CANDIDATES if c['id'] == candidate)
    psan = arm['name'] == 'psan'
    return dict(
        protocol=spec['protocol'], protocol_sha256=sha256(PROTOCOL) if PROTOCOL.exists() else 'unfrozen',
        stage=job['kind'], dataset=job['dataset'], arm=job['arm'], model=arm['name'], seed=job['seed'],
        candidate=candidate, lr=cand['lr'], weight_decay=cand['weight_decay'],
        epochs=spec['screening_epochs'] if job['kind'] == 'screen' else spec['final_epochs'],
        patience=None if job['kind'] == 'screen' else spec['final_patience'],
        warmup_epochs=spec['warmup_epochs'], batch_size=spec['batch_size'], num_workers=spec['num_workers'],
        img_size=size, ema_decay=spec['ema_decay'], gradient_clip=spec['gradient_clip'],
        precision='float32', stochastic_depth=spec['stochastic_depth'], pretrained=False,
        atom='gabor' if psan else '', atoms=job.get('atoms', arm.get('atoms')) if psan else None,
        atom_dropout=arm.get('atom_dropout') if psan else None, init='morlet' if psan else '',
        axis_aligned=False, fixed_phase=False,
        initialization=('non-filter tensors copied from untrained PSAN M16 of the same seed'
                        if job['kind'] == 'ofat' else 'seed_all(seed) then default constructors'),
        num_classes=len(classes), classes=classes, split_sha256=split_sha,
        source_code_sha256=spec['source_code_sha256'], test_used_for_selection=False)


def make_model(job, cfg):
    if cfg['model'] == 'psan':
        model = build_model('psan', classes=cfg['num_classes'], size=cfg['img_size'], atom='gabor',
                            dropout=cfg['atom_dropout'], atoms=cfg['atoms'], init='morlet')
    else:
        model = build_model(cfg['model'], classes=cfg['num_classes'], size=cfg['img_size'])
    if job['kind'] == 'ofat':
        # Same donor rule as experiments.queue_ofat, with the donor seeded by this run's seed.
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(torch.Generator().manual_seed(job['seed']).get_state())
            donor = build_model('psan', classes=cfg['num_classes'], size=cfg['img_size'])
        source, target = donor.state_dict(), model.state_dict()
        copied = 0
        with torch.no_grad():
            for key, value in target.items():
                if '.filter.' not in key and key in source and value.shape == source[key].shape:
                    value.copy_(source[key]); copied += 1
        model.load_state_dict(target)
        print('Controlled initial tensors copied', copied, 'atoms', cfg['atoms'], flush=True)
    return model


def reuse_common_run(job, candidate, directory, cfg):
    path = ROOT / 'results/training' / COMMON_RUNS[job['arm']].format(job['seed']) / 'summary.json'
    result = json.loads(path.read_text())
    previous = json.loads(path.with_name('config.json').read_text())
    expected = dict(lr=1e-4, weight_decay=1e-3, seed=job['seed'], epochs=30, warmup_epochs=3, patience=10,
                    batch_size=128, img_size=224, ema_decay=.999, split_sha256=cfg['split_sha256'])
    if any(previous.get(k) != v for k, v in expected.items()):
        raise ValueError('Common-preset run differs from the c0 final protocol')
    if job['arm'] == 'psan_ad0' and (previous.get('atom_dropout'), previous.get('atoms')) != (0., 16):
        raise ValueError('Common-preset PSAN run is not the AD0 M16 configuration')
    if sha256(ROOT / result['checkpoint']) != result['checkpoint_sha256']:
        raise ValueError('Reused checkpoint hash mismatch')
    summary = dict(result, tag=tag_of(job, candidate), stage='final', candidate=candidate, arm=job['arm'],
                   reused_completed_run=True, reused_summary=str(path.relative_to(ROOT)),
                   protocol_sha256=cfg['protocol_sha256'])
    write_json(directory / 'config.json', dict(cfg, reused_from=str(path.parent.relative_to(ROOT))))
    write_json(directory / 'summary.json', summary)
    return summary


def train(job, device_name, out=OUT, limit=None, epochs=None, workers=None):
    smoke = limit is not None
    spec = protocol_spec() if smoke else load_protocol()
    candidate = resolved_candidate(job, out)
    if candidate is None:
        raise ValueError('Final job has no validation selection yet')
    directory = run_dir(job, candidate, out)
    directory.mkdir(parents=True, exist_ok=True)
    summary_path, config_path = directory / 'summary.json', directory / 'config.json'
    if summary_path.exists():
        result = json.loads(summary_path.read_text())
        if sha256(ROOT / result['checkpoint']) != result['checkpoint_sha256']:
            raise ValueError('Completed round-3 checkpoint changed')
        print('Reusing completed round-3 run', result['tag'], flush=True)
        return result
    splits = ['train', 'val'] if job['kind'] == 'screen' else ['train', 'val', 'test']
    if job['dataset'] == 'canine':
        frames, datasets, classes, split_sha, size = canine_data(splits, limit)
    else:
        frames, datasets, classes, split_sha, size = external_data(job['dataset'], splits, limit)
    cfg = run_config(spec, job, candidate, size, classes, split_sha)
    if epochs:
        cfg['epochs'] = epochs
    if job['kind'] == 'final' and job['dataset'] == 'canine' and candidate == 'c0' and job['arm'] in COMMON_RUNS:
        return reuse_common_run(job, candidate, directory, cfg)
    if config_path.exists() and json.loads(config_path.read_text()) != cfg:
        raise ValueError('Round-3 configuration changed for ' + directory.name)
    write_json(config_path, cfg)
    fingerprint = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    seed_all(cfg['seed'])
    device = torch.device(device_name)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    model = make_model(job, cfg).to(device)
    optimizer = torch.optim.AdamW(build_param_groups(model, cfg['weight_decay'], NO_DECAY), lr=cfg['lr'])
    ema = ModelEMA(model, cfg['ema_decay'], device)
    labels = datasets['train'].labels
    counts = np.bincount(labels, minlength=len(classes))
    sampler_rng = torch.Generator().manual_seed(cfg['seed'] + 1000)
    loader_rng = torch.Generator().manual_seed(cfg['seed'] + 2000)
    sampler = WeightedRandomSampler(torch.as_tensor((1 / np.sqrt(counts))[labels], dtype=torch.double),
                                    len(labels), replacement=True, generator=sampler_rng)
    workers = workers or (2 if smoke else cfg['num_workers'])
    loaders = {s: DataLoader(d, batch_size=cfg['batch_size'], sampler=sampler if s == 'train' else None,
                             shuffle=False, num_workers=workers, pin_memory=device.type == 'cuda',
                             worker_init_fn=seed_worker, generator=loader_rng, persistent_workers=False)
               for s, d in datasets.items()}
    start, best, best_epoch, bad, total, rows = 1, -1., 0, 0, 0., []
    last_path, best_path = directory / 'last.pth', directory / 'best.pth'
    if last_path.exists():
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        if checkpoint['config_fingerprint'] != fingerprint:
            raise ValueError('Round-3 resume fingerprint changed')
        model.load_state_dict(checkpoint['model']); ema.ema.load_state_dict(checkpoint['ema'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start = checkpoint['epoch'] + 1
        best, best_epoch, bad = checkpoint['best_macro_f1'], checkpoint['best_epoch'], checkpoint['patience_counter']
        total, rows, rng = checkpoint['training_time_sec'], checkpoint['history'], checkpoint['rng']
        torch.set_rng_state(rng['torch'].cpu()); np.random.set_state(rng['numpy']); random.setstate(rng['python'])
        sampler_rng.set_state(rng['sampler'].cpu()); loader_rng.set_state(rng['loader'].cpu())
        if device.type == 'cuda':
            torch.cuda.set_rng_state(rng['cuda'].cpu(), device)
        print('Resumed', directory.name, 'at epoch', start, flush=True)
    print('START', directory.name, json.dumps({k: v for k, v in cfg.items() if k != 'classes'}), flush=True)
    for epoch in range(start, cfg['epochs'] + 1):
        if cfg['patience'] is not None and bad >= cfg['patience']:
            break
        tic = time.perf_counter()
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        warmup = cfg['warmup_epochs']
        factor = epoch / warmup if epoch <= warmup else .5 * (1 + math.cos(math.pi * (epoch - warmup) / (cfg['epochs'] - warmup)))
        for group in optimizer.param_groups:
            group['lr'] = cfg['lr'] * factor
        loss, accuracy, extra = train_one_epoch(model, loaders['train'], optimizer, device, ema)
        val_logits, val_y = infer(ema.ema, loaders['val'], device)
        val_f1 = float(f1_score(val_y, val_logits.argmax(-1), labels=range(len(classes)), average='macro', zero_division=0))
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - tic; total += elapsed
        improved = val_f1 > best
        if improved:
            best, best_epoch, bad = val_f1, epoch, 0
        else:
            bad += 1
        rows.append(dict(epoch=epoch, train_loss=loss, train_accuracy=accuracy / 100,
                         validation_macro_f1=val_f1, learning_rate=cfg['lr'] * factor, epoch_time_sec=elapsed,
                         training_time_sec=total, gradient_norm_mean=extra['gradient_norm_mean'],
                         peak_gpu_allocated_mb=torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == 'cuda' else 0))
        checkpoint = dict(epoch=epoch, model=model.state_dict(), ema=ema.ema.state_dict(), eval_model=ema.ema.state_dict(),
                          optimizer=optimizer.state_dict(), config=cfg, config_fingerprint=fingerprint,
                          best_macro_f1=best, best_epoch=best_epoch, patience_counter=bad, training_time_sec=total,
                          history=rows,
                          rng=dict(torch=torch.get_rng_state(), numpy=np.random.get_state(), python=random.getstate(),
                                   sampler=sampler_rng.get_state(), loader=loader_rng.get_state(),
                                   cuda=torch.cuda.get_rng_state(device) if device.type == 'cuda' else None))
        temporary = last_path.with_suffix('.tmp'); torch.save(checkpoint, temporary); temporary.replace(last_path)
        if improved:
            temporary = best_path.with_suffix('.tmp'); torch.save(checkpoint, temporary); temporary.replace(best_path)
        pd.DataFrame(rows).to_csv(directory / 'epochs.csv', index=False)
        print(f'{directory.name} epoch={epoch}/{cfg["epochs"]} val_F1={val_f1:.5f} best={best:.5f} time={elapsed:.1f}s', flush=True)
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['eval_model']); model.eval()
    metrics = {}
    for split in splits[1:]:
        logits, y = infer(model, loaders[split], device)
        prediction_csv(directory / (split + '_predictions.csv'), frames[split], logits, y, classes)
        bundle = compute_metric_bundle(y, torch.as_tensor(logits).softmax(-1).numpy(), classes)
        write_json(directory / (split + '_metrics.json'), bundle)
        metrics[split] = bundle['metrics']
    result = dict(tag=directory.name, job=job, arm=job['arm'], model=cfg['model'], dataset=job['dataset'],
                  seed=cfg['seed'], candidate=candidate, stage=job['kind'], best_validation_epoch=best_epoch,
                  best_validation_macro_f1=best, epochs_completed=rows[-1]['epoch'], max_epochs=cfg['epochs'],
                  training_time_sec=total, peak_gpu_allocated_mb=max(r['peak_gpu_allocated_mb'] for r in rows),
                  checkpoint=str(best_path.relative_to(ROOT) if best_path.is_relative_to(ROOT) else best_path),
                  checkpoint_sha256=sha256(best_path),
                  config_fingerprint=fingerprint, split_sha256=cfg['split_sha256'],
                  protocol_sha256=cfg['protocol_sha256'], parameters=sum(p.numel() for p in model.parameters()),
                  device=device_name, reused_completed_run=False, **metrics)
    write_json(summary_path, result)
    if not smoke:
        last_path.unlink()
    print('COMPLETED', directory.name, 'validation', best, flush=True)
    return result


# ----------------------------------------------------------------------------- scheduling

def select_ready(spec, out=OUT):
    groups = [('canine', a) for a in spec['canine']['screened_arms']]
    groups += [(d, a) for d in spec['external']['datasets'] for a in spec['external']['arms']]
    for dataset, arm in groups:
        target = selection_path(dataset, arm, out)
        if target.exists():
            continue
        paths = [run_dir(dict(kind='screen', dataset=dataset, arm=arm, candidate=c['id'], seed=42), None, out)
                 / 'summary.json' for c in CANDIDATES]
        if not all(p.exists() for p in paths):
            continue
        trials = [json.loads(p.read_text()) for p in paths]
        for trial, path in zip(trials, paths):
            if trial['stage'] != 'screen' or 'test' in trial or any(path.parent.glob('test*')):
                raise ValueError('Selection received a test result')
            if trial['protocol_sha256'] != sha256(PROTOCOL):
                raise ValueError('Selection protocol mismatch')
        winner = max(range(len(trials)), key=lambda i: (trials[i]['best_validation_macro_f1'], -i))
        write_json(target, dict(dataset=dataset, arm=arm, candidate=CANDIDATES[winner],
                                validation_scores=[t['best_validation_macro_f1'] for t in trials],
                                screening_tags=[t['tag'] for t in trials], objective=spec['objective'],
                                protocol_sha256=sha256(PROTOCOL), test_used=False))
        print('VALIDATION SELECTED', dataset, arm, CANDIDATES[winner]['id'], flush=True)


def job_state(job, out=OUT):
    candidate = resolved_candidate(job, out)
    if candidate is None:
        return 'waiting', None
    tag = tag_of(job, candidate)
    if (run_dir(job, candidate, out) / 'summary.json').exists():
        return 'done', tag
    if (out / 'failed' / f'{tag}.json').exists():
        return 'failed', tag
    lock = out / 'locks' / f'{tag}.lock'
    if lock.exists():
        pid = int(json.loads(lock.read_text())['pid'])
        try:
            os.kill(pid, 0)
            return 'running', tag
        except OSError:
            lock.unlink(missing_ok=True)
    if job['dataset'] != 'canine' and not (EXTERNAL_ROOT / f"{job['dataset']}_provenance.json").exists():
        return 'waiting', tag
    return 'pending', tag


def acquire(tag, out=OUT):
    lock = out / 'locks' / f'{tag}.lock'
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        handle = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(handle, 'w') as f:
        json.dump(dict(pid=os.getpid(), started=time.strftime('%Y-%m-%d %H:%M:%S')), f)
    return True


def worker(device):
    spec = freeze()
    logs = OUT / 'logs'; logs.mkdir(parents=True, exist_ok=True)
    while True:
        spec = load_protocol()
        select_ready(spec)
        states = [(job, *job_state(job)) for job in job_list(spec)]
        nxt = next(((job, tag) for job, state, tag in states if state == 'pending' and acquire(tag)), None)
        if nxt is None:
            if all(state in ('done', 'failed') for _, state, _ in states):
                print('All round-3 jobs finished on', device, flush=True)
                return
            time.sleep(60)
            continue
        job, tag = nxt
        print(time.strftime('%H:%M:%S'), device, 'START', tag, flush=True)
        with (logs / f'{tag}.log').open('a') as log:
            code = subprocess.run([sys.executable, '-m', 'experiments.round3', 'job', '--spec', json.dumps(job),
                                   '--device', device], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
        if code != 0:
            write_json(OUT / 'failed' / f'{tag}.json', dict(job=job, device=device, returncode=code,
                                                         log=str((logs / f'{tag}.log').relative_to(ROOT))))
        (OUT / 'locks' / f'{tag}.lock').unlink(missing_ok=True)
        print(time.strftime('%H:%M:%S'), device, 'END', tag, 'code', code, flush=True)


def report():
    spec = load_protocol()
    rows = []
    for job in job_list(spec):
        state, tag = job_state(job)
        rows.append(dict(state=state, tag=tag or f"{job['dataset']}_{job['arm']}_<selection>_s{job['seed']}_final"))
    frame = pd.DataFrame(rows)
    print(frame.state.value_counts().to_string())
    print(frame[frame.state.isin(['running', 'failed'])].to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['freeze', 'prepare', 'worker', 'job', 'status', 'smoke'])
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--spec')
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.command == 'freeze':
        freeze(); print('Frozen', sha256(PROTOCOL))
    elif args.command == 'prepare':
        prepare_external()
    elif args.command == 'worker':
        worker(args.device)
    elif args.command == 'job':
        train(json.loads(args.spec), args.device)
    elif args.command == 'status':
        report()
    elif args.command == 'smoke':
        scratch = Path(os.environ['ROUND3_SMOKE_DIR'])
        jobs = [dict(kind='screen', dataset='canine', arm='gfnet_matched', candidate='c2', seed=42),
                dict(kind='final', dataset='canine', arm='psan', candidate='c2', seed=43),
                dict(kind='ofat', dataset='canine', arm='psan_ad0_M32', candidate='c0', seed=43, atoms=32)]
        jobs += [dict(kind='final', dataset=d, arm=a, candidate='c2', seed=42)
                 for d in EXTERNAL if (EXTERNAL_ROOT / f'{d}_provenance.json').exists()
                 for a in ['psan', 'gfnet_matched', 'gfnet', 'afno']]
        for job in jobs:
            result = train(job, args.device, out=scratch, limit=300, epochs=1)
            print('SMOKE OK', result['tag'], result['parameters'], result.get('test', {}).get('macro_f1'), flush=True)


if __name__ == '__main__':
    main()
