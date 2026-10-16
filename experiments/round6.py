"""Round-6 follow-up (2026-10-15): nested cross-validation, disease-only training, 100-epoch replication and a
token-grid ablation for PSAN, PSAN without atom dropout, the matched dense control and the free rank-16 control.

A second external review of the MDPI draft asked (i) for a cross-validation in which the training setting of every
outer fold is selected without that fold's test images, (ii) for models trained without the Normal class, which is
confounded with the smartphone device, (iii) for the 100-epoch comparison over several seeds and with the free
low-rank control, and (iv) for a test of whether the 8x8 token grid of the MedMNIST+ benchmarks, rather than the data,
explains why the atom advantage did not transfer. Outputs are written under results/round6/; earlier rounds are
unchanged. The epoch loop is copied from experiments.round5.train (itself copied from rounds 3-4) without modification
of the optimization; only the data, the image size, the arms and the output directories differ. All earlier test
results were known when this protocol was frozen; it is a disclosed follow-up, not a blinded holdout.

Experiments (seed 42 for screening; c0-c2 screened for eight epochs on validation Macro F1 only):
  ncv   outer folds 2-5: re-screen the four arms on each fold's own training and validation folds. If a fold selects
        c2, the setting of its round-4 cross-validation final, that run is reused unchanged; otherwise the selected
        setting is trained (seed 42, at most 30 epochs). Fold 1 is the main split, whose selection used only its own
        validation fold, so the round-3/4 seed-42 finals serve as fold 1.
  dis   ten disease classes only (Normal images removed from training, validation and test): screening, then the
        selected setting at seeds 42-44.
  long  100 epochs with c2 and no early stopping at seeds 43-44 (PSAN, dense) and 42-44 (free rank-16); the seed-42
        PSAN and dense runs of round 4 complete three seeds.
  g128  128-pixel inputs (8x8 token grid, 40 stored frequency bins as in MedMNIST+128), PSAN and the dense control:
        screening, then the selected setting at seeds 42-44.

    $PY -m experiments.round6 freeze
    ROUND6_SCRATCH=<dir> $PY -m experiments.round6 smoke --device cuda:0
    ROUND6_SCRATCH=<dir> $PY -m experiments.round6 replay --device cuda:0   # first-epoch reproduction of round-4 runs
    $PY -m experiments.round6 worker --device cuda:0
    $PY -m experiments.round6 status
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
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms

from .common import (CLASSES, MEAN, ROOT, STD, ManifestDataset, build_model, infer, load_manifest, prediction_csv,
                     seed_all, sha256, write_json)
from .training import ModelEMA, build_param_groups, seed_worker, train_one_epoch
from evaluation_metrics import compute_metric_bundle
from models.lowrank import to_low_rank

OUT = ROOT / 'results/round6'
PROTOCOL = ROOT / 'configs/round6_protocol.json'
SOURCES = ['experiments/round6.py', 'experiments/common.py', 'experiments/training.py', 'models/psan.py',
           'models/gfnet.py', 'models/lowrank.py', 'train/train_base.py', 'evaluation_metrics.py']
NO_DECAY = ('norm', 'filter_real', 'filter_imag', 'mu', 'log_sigma', 'rho_raw',
            'theta', 'log_amp', 'phi', 'w_real', 'w_imag', 'basis_real', 'basis_imag')
CANDIDATES = [dict(id='c0', lr=1e-4, weight_decay=1e-3),
              dict(id='c1', lr=3e-4, weight_decay=1e-3),
              dict(id='c2', lr=5e-4, weight_decay=5e-2)]
SEEDS = [42, 43, 44]
ARMS = {'psan': dict(name='psan', atom_dropout=.2, atoms=16),
        'psan_ad0': dict(name='psan', atom_dropout=0., atoms=16),
        'gfnet_matched': dict(name='gfnet_matched'),
        'lowrank16': dict(name='lowrank', atoms=16)}
NCV_ARMS = ['psan', 'psan_ad0', 'gfnet_matched', 'lowrank16']
NCV_FOLDS = [2, 3, 4, 5]
NCV_REUSE = dict(candidate='c2', directory='results/round4/cv/canine_{arm}_c2_s42_cv{fold}')
DIS_ARMS = ['psan', 'gfnet_matched', 'lowrank16']
LONG_RUNS = [('psan', 43), ('psan', 44), ('gfnet_matched', 43), ('gfnet_matched', 44),
             ('lowrank16', 42), ('lowrank16', 43), ('lowrank16', 44)]
G128_ARMS = ['psan', 'gfnet_matched']
DISEASES = [c for c in CLASSES if c != 'Normal']
GROUPS = {'ncv': 'ncv', 'dis': 'disease', 'long': 'long', 'g128': 'grid128'}
# Replay only: earlier runs whose first epoch must be reproduced exactly by this code before launch.
REPLAY = [dict(exp='ncv', kind='final', arm='psan', candidate='c2', seed=42, fold=3,
               reference='results/round4/cv/canine_psan_c2_s42_cv3'),
          dict(exp='long', kind='final', arm='lowrank16', candidate='c2', seed=42,
               reference='results/round4/final/canine_lowrank16_c2_s42_final'),
          dict(exp='long', kind='final', arm='psan', candidate='c2', seed=42,
               reference='results/round4/long/canine_psan_c2_s42_long100')]


def outer_folds():
    folds = json.loads((ROOT / 'configs/outer_cv.json').read_text())['folds']
    return {f['test']: f for f in folds}


def cv_split(fold):
    frame = load_manifest().copy()
    spec = outer_folds()[fold]
    frame['split'] = np.where(frame.fold == spec['test'], 'test', np.where(frame.fold == spec['validation'], 'val', 'train'))
    if (frame.groupby('group_id').split.nunique() > 1).any():
        raise ValueError('Visual group crosses CV partitions')
    return frame


def split_digest(frame, columns=('path', 'split')):
    return hashlib.sha256(frame[list(columns)].to_csv(index=False).encode()).hexdigest()


def disease_frame():
    frame = load_manifest()
    frame = frame[frame['class'] != 'Normal'].copy()
    frame['label'] = frame['class'].map({c: i for i, c in enumerate(DISEASES)}).astype(int)
    return frame.reset_index(drop=True)


def protocol_spec():
    folds = outer_folds()
    dis = disease_frame()
    return dict(
        protocol='round6_nested_cv_disease_long_grid_v1',
        frozen_on='2026-10-15',
        motivation=('Second external review: nested selection in the cross-validation, training without the '
                    'device-confounded Normal class, a multi-seed 100-epoch comparison including the free low-rank '
                    'control, and a token-grid ablation separating grid size from dataset.'),
        prior_exposure=['all round-2 to round-5 test results, including the round-4 cross-validation, the seed-42 '
                        '100-epoch runs, the disease-only evaluation of the 11-class models and the MedMNIST+128 '
                        'benchmarks'],
        not_a_blinded_holdout=True,
        arms=ARMS,
        screening=dict(seed=42, candidates=CANDIDATES, epochs=8, early_stopping=False,
                       objective='highest best-epoch validation Macro F1; ties use earlier candidate order'),
        nested_cv=dict(
            arms=NCV_ARMS, folds=[folds[i] for i in NCV_FOLDS], seed=42, epochs=30, patience=10,
            selection='re-screened on the training and validation folds of each outer fold; its test fold unused',
            reuse=('a fold that selects c2 reuses the round-4 cross-validation final of that arm and fold, which was '
                   'trained with c2, seed 42 and the identical pipeline; other selections are trained here'),
            reuse_directory=NCV_REUSE['directory'],
            fold1=('the validation-selected seed-42 finals on the main split; their selection used only training folds '
                   '3-5 and validation fold 2'),
            split_sha256={str(f): split_digest(cv_split(f)) for f in NCV_FOLDS}),
        disease_only=dict(
            arms=DIS_ARMS, classes=DISEASES, seeds=SEEDS, epochs=30, patience=10,
            data='main split with every Normal image removed from training, validation and test; labels re-indexed',
            images={s: int((dis.split == s).sum()) for s in ['train', 'val', 'test']},
            split_sha256=split_digest(dis, ('path', 'split', 'label'))),
        long_training=dict(
            runs=[dict(arm=a, seed=s) for a, s in LONG_RUNS], candidate='c2', epochs=100, warmup_epochs=3,
            early_stopping=False, selection='best validation Macro F1 epoch, as in all rounds',
            seed42_reference=['results/round4/long/canine_psan_c2_s42_long100',
                              'results/round4/long/canine_gfnet_matched_c2_s42_long100']),
        grid128=dict(
            arms=G128_ARMS, seeds=SEEDS, epochs=30, patience=10, img_size=128, token_grid='8x8, 40 stored bins',
            resize=('every image is resized to 224x224 (the cached 224-pixel image where available, as in all canine '
                    'runs) and then to 128x128 with the same bilinear interpolation'),
            filter_parameters_per_block=dict(psan=6272, gfnet_matched=15360)),
        common=dict(batch_size=128, num_workers=6, ema_decay=.999, gradient_clip=1., warmup_epochs=3,
                    stochastic_depth=.1, sampler='inverse-square-root class-weighted replacement sampling',
                    precision='float32 tensors; CUDA matmul TF32 off; cuDNN convolution TF32 default',
                    no_weight_decay=list(NO_DECAY)),
        forbidden_selection_inputs=['test images', 'test labels', 'test predictions or metrics'],
        statistics=dict(
            bootstrap=10000,
            ci='paired percentile bootstrap over test visual groups; crossed group x seed bootstrap across seeds 42-44',
            correction='Holm within each declared family',
            families=dict(
                nested_cv=['psan minus gfnet_matched', 'psan_ad0 minus gfnet_matched', 'psan minus lowrank16',
                           'psan_ad0 minus lowrank16'],
                disease_only=['psan minus gfnet_matched', 'psan minus lowrank16'],
                long_training=['psan minus gfnet_matched', 'psan minus lowrank16'],
                grid128=['psan minus gfnet_matched']),
            nested_cv_summary='pooled out-of-fold Macro F1 over all 104,037 images with a paired bootstrap over all '
                              'visual groups, and per-fold paired bootstraps',
            disease_only_metric='Macro F1 over the ten disease classes; device-restricted rows are descriptive'),
        replay=[{k: r[k] for k in ['exp', 'arm', 'candidate', 'seed', 'reference']} for r in REPLAY],
        job_order='disease and grid screening, their finals, nested-CV screening and finals, then the 100-epoch runs',
        checkpoint_retention='best.pth kept; resume-only last.pth deleted after a completed run',
        concurrency='up to two jobs per GPU, shared with other users; wall-clock times are not reported as cost',
        source_code_sha256={p: sha256(ROOT / p) for p in SOURCES})


def freeze():
    spec = protocol_spec()
    if PROTOCOL.exists():
        if json.loads(PROTOCOL.read_text()) != spec:
            raise ValueError('Frozen round-6 protocol or source differs; create a new protocol version')
    else:
        write_json(PROTOCOL, spec)
    return spec


def load_protocol():
    spec = json.loads(PROTOCOL.read_text())
    if spec != protocol_spec():
        raise ValueError('Round-6 source/protocol changed after freezing')
    return spec


# ----------------------------------------------------------------------------- jobs

def tag_of(job, candidate=None):
    cid = candidate or job['candidate']
    stage = 'screening' if job['kind'] == 'screen' else 'final'
    if job['exp'] == 'ncv':
        return f"canine_{job['arm']}_{cid}_s{job['seed']}_ncv{job['fold']}" + ('_screening' if job['kind'] == 'screen' else '')
    if job['exp'] == 'long':
        return f"canine_{job['arm']}_{cid}_s{job['seed']}_long100"
    prefix = {'dis': 'canine10', 'g128': 'canine128'}[job['exp']]
    return f"{prefix}_{job['arm']}_{cid}_s{job['seed']}_{stage}"


def run_dir(job, candidate=None, out=OUT):
    group = out / GROUPS[job['exp']]
    if job['exp'] != 'long':
        group = group / ('trials' if job['kind'] == 'screen' else 'final')
    return group / tag_of(job, candidate)


def selection_key(job):
    return f"{job['exp']}_{job['arm']}" + (f"_fold{job['fold']}" if job['exp'] == 'ncv' else '')


def selection_path(job, out=OUT):
    return out / 'selections' / f'{selection_key(job)}.json'


def resolved_candidate(job, out=OUT):
    if job['candidate'] is not None:
        return job['candidate']
    path = selection_path(job, out)
    return json.loads(path.read_text())['candidate']['id'] if path.exists() else None


def reused_dir(job, candidate):
    if job['exp'] == 'ncv' and job['kind'] == 'final' and candidate == NCV_REUSE['candidate']:
        return ROOT / NCV_REUSE['directory'].format(arm=job['arm'], fold=job['fold'])
    return None


def job_list(spec=None):
    screen = lambda exp, arms, **extra: [dict(exp=exp, kind='screen', arm=a, candidate=c['id'], seed=42, **extra)
                                         for a in arms for c in CANDIDATES]
    jobs = screen('dis', DIS_ARMS) + screen('g128', G128_ARMS)
    jobs += [dict(exp='dis', kind='final', arm=a, candidate=None, seed=s) for a in DIS_ARMS for s in SEEDS]
    jobs += [dict(exp='g128', kind='final', arm=a, candidate=None, seed=s) for a in G128_ARMS for s in SEEDS]
    for fold in NCV_FOLDS:
        jobs += screen('ncv', NCV_ARMS, fold=fold)
    jobs += [dict(exp='ncv', kind='final', arm=a, candidate=None, seed=42, fold=f) for f in NCV_FOLDS for a in NCV_ARMS]
    jobs += [dict(exp='long', kind='final', arm=a, candidate='c2', seed=s) for a, s in LONG_RUNS]
    return jobs


# ----------------------------------------------------------------------------- training

def transform128(train):
    items = [transforms.Resize((224, 224)), transforms.Resize((128, 128))]
    if train:
        items += [transforms.RandomHorizontalFlip(), transforms.ColorJitter(.15, .15, .10, .05)]
    return transforms.Compose(items + [transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def job_data(job, splits, limit=None):
    if job['exp'] == 'ncv':
        frame, classes = cv_split(job['fold']), CLASSES
        digest = split_digest(frame)
    elif job['exp'] == 'dis':
        frame, classes = disease_frame(), DISEASES
        digest = split_digest(frame, ('path', 'split', 'label'))
    else:
        frame, classes = load_manifest(), CLASSES
        digest = sha256(ROOT / 'results/revision_split.csv')
    frames = {s: frame[frame.split == s].reset_index(drop=True) for s in splits}
    if limit:
        frames = {s: f.iloc[:limit].reset_index(drop=True) for s, f in frames.items()}
    datasets = {s: ManifestDataset(f, s == 'train', 224) for s, f in frames.items()}
    if job['exp'] == 'g128':
        for s, d in datasets.items():
            d.transform = transform128(s == 'train')
    return frames, datasets, classes, digest


def run_config(spec, job, candidate, classes, split_sha):
    arm = ARMS[job['arm']]
    cand = next(c for c in CANDIDATES if c['id'] == candidate)
    epochs = 8 if job['kind'] == 'screen' else 100 if job['exp'] == 'long' else 30
    patience = None if job['kind'] == 'screen' or job['exp'] == 'long' else 10
    common = spec['common']
    return dict(
        protocol=spec['protocol'], protocol_sha256=sha256(PROTOCOL) if PROTOCOL.exists() else 'unfrozen',
        experiment=job['exp'], stage=job['kind'], dataset={'dis': 'canine10'}.get(job['exp'], 'canine'),
        arm=job['arm'], model=arm['name'], seed=job['seed'], fold=job.get('fold', 1), candidate=candidate,
        lr=cand['lr'], weight_decay=cand['weight_decay'], epochs=epochs, patience=patience,
        warmup_epochs=common['warmup_epochs'], batch_size=common['batch_size'], num_workers=common['num_workers'],
        img_size=128 if job['exp'] == 'g128' else 224, ema_decay=common['ema_decay'],
        gradient_clip=common['gradient_clip'], precision='float32', stochastic_depth=common['stochastic_depth'],
        pretrained=False, atoms=arm.get('atoms'),
        atom_dropout=arm.get('atom_dropout', 0. if arm['name'] != 'gfnet_matched' else None),
        num_classes=len(classes), classes=classes, split_sha256=split_sha,
        source_code_sha256=spec['source_code_sha256'], test_used_for_selection=False)


def make_model(cfg):
    name = cfg['model']
    if name == 'gfnet_matched':
        return build_model('gfnet_matched', classes=cfg['num_classes'], size=cfg['img_size'])
    dropout = cfg['atom_dropout'] if name == 'psan' else 0.
    model = build_model('psan', classes=cfg['num_classes'], size=cfg['img_size'], atom='gabor', dropout=dropout,
                        atoms=16, init='morlet')
    if name == 'lowrank':
        to_low_rank(model, cfg['atoms'])
    elif name != 'psan':
        raise ValueError(name)
    return model


def train(job, device_name, out=OUT, limit=None, epochs=None, workers=None):
    smoke = limit is not None or epochs is not None
    spec = protocol_spec() if smoke else load_protocol()
    candidate = resolved_candidate(job, out)
    if candidate is None:
        raise ValueError('Job has no validation selection yet')
    if not smoke and reused_dir(job, candidate) is not None:
        raise ValueError('This nested-CV final reuses a round-4 run and is not trained again')
    directory = run_dir(job, candidate, out)
    directory.mkdir(parents=True, exist_ok=True)
    summary_path, config_path = directory / 'summary.json', directory / 'config.json'
    if summary_path.exists():
        result = json.loads(summary_path.read_text())
        if sha256(ROOT / result['checkpoint']) != result['checkpoint_sha256']:
            raise ValueError('Completed round-6 checkpoint changed')
        print('Reusing completed round-6 run', result['tag'], flush=True)
        return result
    splits = ['train', 'val'] if job['kind'] == 'screen' else ['train', 'val', 'test']
    frames, datasets, classes, split_sha = job_data(job, splits, limit)
    cfg = run_config(spec, job, candidate, classes, split_sha)
    if epochs:
        cfg['epochs'] = epochs
    if config_path.exists() and json.loads(config_path.read_text()) != cfg:
        raise ValueError('Round-6 configuration changed for ' + directory.name)
    write_json(config_path, cfg)
    fingerprint = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    seed_all(cfg['seed'])
    device = torch.device(device_name)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    model = make_model(cfg).to(device)
    optimizer = torch.optim.AdamW(build_param_groups(model, cfg['weight_decay'], NO_DECAY), lr=cfg['lr'])
    ema = ModelEMA(model, cfg['ema_decay'], device)
    labels = datasets['train'].labels
    counts = np.bincount(labels, minlength=len(classes))
    sampler_rng = torch.Generator().manual_seed(cfg['seed'] + 1000)
    loader_rng = torch.Generator().manual_seed(cfg['seed'] + 2000)
    sampler = WeightedRandomSampler(torch.as_tensor((1 / np.sqrt(counts))[labels], dtype=torch.double),
                                    len(labels), replacement=True, generator=sampler_rng)
    workers = workers or (2 if limit else cfg['num_workers'])
    loaders = {s: DataLoader(d, batch_size=cfg['batch_size'], sampler=sampler if s == 'train' else None,
                             shuffle=False, num_workers=workers, pin_memory=device.type == 'cuda',
                             worker_init_fn=seed_worker, generator=loader_rng, persistent_workers=False)
               for s, d in datasets.items()}
    start, best, best_epoch, bad, total, rows = 1, -1., 0, 0, 0., []
    last_path, best_path = directory / 'last.pth', directory / 'best.pth'
    if last_path.exists():
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        if checkpoint['config_fingerprint'] != fingerprint:
            raise ValueError('Round-6 resume fingerprint changed')
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
    result = dict(tag=directory.name, job=job, experiment=job['exp'], arm=job['arm'], model=cfg['model'],
                  dataset=cfg['dataset'], seed=cfg['seed'], fold=cfg['fold'], candidate=candidate, stage=job['kind'],
                  img_size=cfg['img_size'], classes=classes,
                  best_validation_epoch=best_epoch, best_validation_macro_f1=best, epochs_completed=rows[-1]['epoch'],
                  max_epochs=cfg['epochs'], training_time_sec=total,
                  peak_gpu_allocated_mb=max(r['peak_gpu_allocated_mb'] for r in rows),
                  checkpoint=str(best_path.relative_to(ROOT) if best_path.is_relative_to(ROOT) else best_path),
                  checkpoint_sha256=sha256(best_path), config_fingerprint=fingerprint, split_sha256=cfg['split_sha256'],
                  protocol_sha256=cfg['protocol_sha256'], parameters=sum(p.numel() for p in model.parameters()),
                  trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  device=device_name, **metrics)
    write_json(summary_path, result)
    if not smoke:
        last_path.unlink()
    print('COMPLETED', directory.name, 'validation', best, flush=True)
    return result


def replay(device, scratch):
    """Re-run the first epoch of earlier runs with this code and compare with their logs."""
    rows = []
    for item in REPLAY:
        job = {k: item[k] for k in ['exp', 'kind', 'arm', 'candidate', 'seed'] + (['fold'] if 'fold' in item else [])}
        result = train(job, device, out=Path(scratch), epochs=1)
        checkpoint = Path(result['checkpoint'])
        new = pd.read_csv((checkpoint if checkpoint.is_absolute() else ROOT / checkpoint).parent / 'epochs.csv').iloc[0]
        old = pd.read_csv(ROOT / item['reference'] / 'epochs.csv').iloc[0]
        rows.append(dict(reference=item['reference'], train_loss_new=float(new.train_loss), train_loss_old=float(old.train_loss),
                         val_f1_new=float(new.validation_macro_f1), val_f1_old=float(old.validation_macro_f1),
                         agree_6dp=bool(round(new.train_loss, 6) == round(old.train_loss, 6) and
                                        round(new.validation_macro_f1, 6) == round(old.validation_macro_f1, 6))))
        print('REPLAY', json.dumps(rows[-1]), flush=True)
    write_json(OUT / 'replay_first_epoch.json', dict(rows=rows, device=device, note='epoch-1 replay with round-6 code'))
    if not all(r['agree_6dp'] for r in rows):
        raise SystemExit('Round-6 replay did not reproduce the earlier first epochs')


# ----------------------------------------------------------------------------- scheduling

def select_ready(spec, out=OUT):
    units = {}
    for job in job_list(spec):
        if job['kind'] == 'screen':
            units.setdefault(selection_key(job), []).append(job)
    for key, jobs in units.items():
        target = selection_path(jobs[0], out)
        if target.exists():
            continue
        paths = [run_dir(j, None, out) / 'summary.json' for j in jobs]
        if not all(p.exists() for p in paths):
            continue
        trials = [json.loads(p.read_text()) for p in paths]
        for trial, path in zip(trials, paths):
            if trial['stage'] != 'screen' or 'test' in trial or any(path.parent.glob('test*')):
                raise ValueError('Selection received a test result')
            if trial['protocol_sha256'] != sha256(PROTOCOL):
                raise ValueError('Selection protocol mismatch')
        winner = max(range(len(trials)), key=lambda i: (trials[i]['best_validation_macro_f1'], -i))
        write_json(target, dict(unit=key, experiment=jobs[0]['exp'], arm=jobs[0]['arm'], fold=jobs[0].get('fold'),
                                candidate=CANDIDATES[winner],
                                validation_scores=[t['best_validation_macro_f1'] for t in trials],
                                screening_tags=[t['tag'] for t in trials], objective=spec['screening']['objective'],
                                protocol_sha256=sha256(PROTOCOL), test_used=False))
        print('VALIDATION SELECTED', key, CANDIDATES[winner]['id'], flush=True)


def job_state(job, out=OUT):
    candidate = resolved_candidate(job, out)
    if candidate is None:
        return 'waiting', None
    tag = tag_of(job, candidate)
    reused = reused_dir(job, candidate)
    if reused is not None:
        if not (reused / 'summary.json').exists():
            raise ValueError('Missing round-4 run for reuse: ' + str(reused))
        return 'reused', tag
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
    logs = OUT / 'logs'; logs.mkdir(parents=True, exist_ok=True)
    while True:
        spec = load_protocol()
        select_ready(spec)
        states = [(job, *job_state(job)) for job in job_list(spec)]
        nxt = next(((job, tag) for job, state, tag in states if state == 'pending' and acquire(tag)), None)
        if nxt is None:
            if all(state in ('done', 'reused', 'failed') for _, state, _ in states):
                print('All round-6 jobs finished on', device, flush=True)
                return
            time.sleep(60)
            continue
        job, tag = nxt
        print(time.strftime('%H:%M:%S'), device, 'START', tag, flush=True)
        with (logs / f'{tag}.log').open('a') as log:
            code = subprocess.run([sys.executable, '-m', 'experiments.round6', 'job', '--spec', json.dumps(job),
                                   '--device', device], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
        if code != 0:
            write_json(OUT / 'failed' / f'{tag}.json', dict(job=job, device=device, returncode=code,
                                                         log=str((logs / f'{tag}.log').relative_to(ROOT))))
        (OUT / 'locks' / f'{tag}.lock').unlink(missing_ok=True)
        print(time.strftime('%H:%M:%S'), device, 'END', tag, 'code', code, flush=True)


def report():
    spec = load_protocol()
    rows, seen = [], set()
    for job in job_list(spec):
        state, tag = job_state(job)
        tag = tag or tag_of(job, '<selection>')
        if tag in seen:
            continue
        seen.add(tag)
        rows.append(dict(experiment=job['exp'], state=state, tag=tag))
    frame = pd.DataFrame(rows)
    print(frame.groupby(['experiment', 'state']).size().to_string())
    print(frame[frame.state.isin(['running', 'failed'])].to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['freeze', 'worker', 'job', 'status', 'smoke', 'replay'])
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--spec')
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.command == 'freeze':
        freeze(); print('Frozen', sha256(PROTOCOL))
    elif args.command == 'worker':
        worker(args.device)
    elif args.command == 'job':
        train(json.loads(args.spec), args.device)
    elif args.command == 'status':
        report()
    elif args.command == 'replay':
        replay(args.device, os.environ['ROUND6_SCRATCH'])
    elif args.command == 'smoke':
        scratch = Path(os.environ['ROUND6_SCRATCH'])
        tests = [dict(exp='dis', kind='final', arm='psan', candidate='c2', seed=42),
                 dict(exp='dis', kind='final', arm='lowrank16', candidate='c2', seed=42),
                 dict(exp='g128', kind='final', arm='psan', candidate='c2', seed=42),
                 dict(exp='g128', kind='final', arm='gfnet_matched', candidate='c2', seed=42),
                 dict(exp='ncv', kind='final', arm='psan_ad0', candidate='c0', seed=42, fold=4),
                 dict(exp='long', kind='final', arm='lowrank16', candidate='c2', seed=43)]
        for job in tests:
            result = train(job, args.device, out=scratch, limit=256, epochs=1)
            print('SMOKE OK', result['tag'], result['img_size'], len(result['classes']), result['parameters'],
                  round(result['test']['macro_f1'], 4), flush=True)


if __name__ == '__main__':
    main()
