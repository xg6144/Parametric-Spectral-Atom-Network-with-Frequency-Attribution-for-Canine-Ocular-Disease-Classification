"""Round-5 follow-up (2026-10-15): five general-purpose vision backbones trained from scratch.

The article compares architectures trained from scratch only. This round adds DeiT-Tiny, ConvNeXt-Atto, FastViT-T8,
Swin-Tiny and SHViT-S1 (timm definitions, random initialization, no pretrained weights) under the two canine protocols
of the article: the shared preset (candidate c0: learning rate 1e-4, weight decay 1e-3, at most 30 epochs, patience 10)
and validation-selected training (eight-epoch screening of c0-c2 at seed 42 on validation Macro F1, then the selected
setting for at most 30 epochs). Both protocols use seeds 42-44. Because the shared preset equals c0 trained as a final,
an arm that selects c0 reuses its shared-preset runs. Outputs are written under results/round5/; earlier rounds are
unchanged. The epoch loop is copied from experiments.round4.train without modification of the optimization; only the
model constructor, the arms and the output directory differ. All earlier test results were known when this protocol
was frozen; it is a disclosed follow-up, not a blinded holdout.

    $PY -m experiments.round5 freeze
    ROUND5_SCRATCH=<dir> $PY -m experiments.round5 smoke --device cuda:0
    ROUND5_SCRATCH=<dir> $PY -m experiments.round5 replay --device cuda:0   # first-epoch reproduction of earlier runs
    $PY -m experiments.round5 worker --device cuda:0
    $PY -m experiments.round5 status
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

from .common import CLASSES, ROOT, ManifestDataset, build_model, infer, load_manifest, prediction_csv, seed_all, sha256, write_json
from .training import ModelEMA, build_param_groups, seed_worker, train_one_epoch
from evaluation_metrics import compute_metric_bundle

OUT = ROOT / 'results/round5'
PROTOCOL = ROOT / 'configs/round5_protocol.json'
SOURCES = ['experiments/round5.py', 'experiments/common.py', 'experiments/training.py', 'models/psan.py',
           'models/gfnet.py', 'train/train_base.py', 'evaluation_metrics.py']
NO_DECAY = ('norm', 'filter_real', 'filter_imag', 'mu', 'log_sigma', 'rho_raw',
            'theta', 'log_amp', 'phi', 'w_real', 'w_imag', 'basis_real', 'basis_imag')
CANDIDATES = [dict(id='c0', lr=1e-4, weight_decay=1e-3),
              dict(id='c1', lr=3e-4, weight_decay=1e-3),
              dict(id='c2', lr=5e-4, weight_decay=5e-2)]
SHARED = 'c0'
SEEDS = [42, 43, 44]
# Arm key -> constructor name of experiments.common.build_model (timm definitions for the backbones).
ARMS = {'deit_scratch': dict(name='deit', timm='deit_tiny_patch16_224'),
        'convnext_scratch': dict(name='convnext', timm='convnext_atto'),
        'fastvit_t8_scratch': dict(name='fastvit_t8', timm='fastvit_t8'),
        'swin_scratch': dict(name='swin', timm='swin_tiny_patch4_window7_224'),
        'shvit_s1_scratch': dict(name='shvit_s1', timm='shvit_s1')}
# Replay only: earlier runs whose first epoch must be reproduced exactly by this code before launch.
REPLAY_ARMS = {'gfnet_matched': dict(name='gfnet_matched'), 'convnext_imagenet': dict(name='convnext', pretrained=True)}
REPLAY = [dict(kind='final', arm='gfnet_matched', candidate='c2', seed=42,
               reference='results/round3/tuning/final/canine_gfnet_matched_c2_s42_final'),
          dict(kind='screen', arm='convnext_imagenet', candidate='c2', seed=42,
               reference='results/tuning/trials/canine_convnext_c2_s42_screening')]


def protocol_spec():
    return dict(
        protocol='round5_scratch_backbones_v1',
        frozen_on='2026-10-15',
        motivation=('The article compares models trained from scratch only. Five general-purpose backbones that had '
                    'been evaluated only with ImageNet weights are trained from random initialization under the '
                    'protocols of the scratch models, so that PSAN is compared with them on equal terms.'),
        prior_exposure=['all round-2 to round-4 test results, including those of the same five architectures '
                        'fine-tuned from ImageNet weights'],
        not_a_blinded_holdout=True,
        arms=ARMS,
        initialization='random (timm default initialization under seed_all(seed)); pretrained=False; no weights loaded',
        shared_preset=dict(arms=list(ARMS), candidate=SHARED, seeds=SEEDS, epochs=30, patience=10,
                           note='identical to the shared preset of PSAN, GFNet and AFNO (c0 trained as a final)'),
        screening=dict(arms=list(ARMS), seed=42, candidates=CANDIDATES, epochs=8, early_stopping=False,
                       objective='highest best-epoch validation Macro F1; ties use earlier candidate order'),
        finals=dict(arms=list(ARMS), seeds=SEEDS, epochs=30, patience=10,
                    note='an arm that selects c0 reuses its shared-preset runs (identical configuration)'),
        common=dict(batch_size=128, num_workers=6, ema_decay=.999, gradient_clip=1., warmup_epochs=3,
                    stochastic_depth='0.1 (SHViT-S1 has no stochastic-depth argument in this timm version)',
                    sampler='inverse-square-root class-weighted replacement sampling',
                    precision='float32 tensors; CUDA matmul TF32 off; cuDNN convolution TF32 default',
                    no_weight_decay=list(NO_DECAY) + ['every parameter with ndim <= 1 and every bias']),
        forbidden_selection_inputs=['test images', 'test labels', 'test predictions or metrics'],
        reference_runs=dict(
            shared='shared-preset PSAN (atom dropout 0.2), seeds 42-44 of results/training',
            tuned='validation-selected PSAN (atom dropout 0.2, selected c2), seeds 42-44 of rounds 2-3'),
        statistics=dict(
            bootstrap=10000,
            ci='paired percentile bootstrap over test visual groups; crossed group x seed bootstrap across seeds 42-44',
            correction='Holm within each declared family',
            families=dict(
                tuned_backbones=[f'tuned psan minus {a}' for a in ARMS],
                shared_backbones=[f'psan minus {a}' for a in ARMS]),
            primary='tuned_backbones', secondary='shared_backbones'),
        replay=[{k: r[k] for k in ['arm', 'candidate', 'seed', 'reference']} for r in REPLAY],
        checkpoint_retention='best.pth kept; resume-only last.pth deleted after a completed run',
        concurrency='up to two jobs per GPU, shared with other users; wall-clock times are reported only as observed',
        source_code_sha256={p: sha256(ROOT / p) for p in SOURCES})


def freeze():
    spec = protocol_spec()
    if PROTOCOL.exists():
        if json.loads(PROTOCOL.read_text()) != spec:
            raise ValueError('Frozen round-5 protocol or source differs; create a new protocol version')
    else:
        write_json(PROTOCOL, spec)
    return spec


def load_protocol():
    spec = json.loads(PROTOCOL.read_text())
    if spec != protocol_spec():
        raise ValueError('Round-5 source/protocol changed after freezing')
    return spec


# ----------------------------------------------------------------------------- jobs

def tag_of(job, candidate=None):
    cid = candidate or job['candidate']
    suffix = {'screen': 'screening', 'final': 'final'}[job['kind']]
    return f"canine_{job['arm']}_{cid}_s{job['seed']}_{suffix}"


def run_dir(job, candidate=None, out=OUT):
    stage = {'screen': 'trials', 'final': 'final'}[job['kind']]
    return out / stage / tag_of(job, candidate)


def selection_path(arm, out=OUT):
    return out / 'selections' / f'canine_{arm}.json'


def resolved_candidate(job, out=OUT):
    if job['candidate'] is not None:
        return job['candidate']
    path = selection_path(job['arm'], out)
    return json.loads(path.read_text())['candidate']['id'] if path.exists() else None


def job_list(spec):
    # Screening and the shared preset first; validation-selected finals become runnable after selection.
    jobs = [dict(kind='screen', arm=a, candidate=c['id'], seed=42) for a in ARMS for c in CANDIDATES]
    jobs += [dict(kind='final', arm=a, candidate=SHARED, seed=s) for a in ARMS for s in spec['shared_preset']['seeds']]
    jobs += [dict(kind='final', arm=a, candidate=None, seed=s) for a in ARMS for s in spec['finals']['seeds']]
    return jobs


# ----------------------------------------------------------------------------- training

def canine_data(splits, limit=None):
    frame = load_manifest()
    frames = {s: frame[frame.split == s].reset_index(drop=True) for s in splits}
    if limit:
        frames = {s: f.iloc[:limit].reset_index(drop=True) for s, f in frames.items()}
    datasets = {s: ManifestDataset(f, s == 'train', 224) for s, f in frames.items()}
    return frames, datasets, CLASSES, sha256(ROOT / 'results/revision_split.csv')


def run_config(spec, job, candidate, classes, split_sha):
    arm = ARMS.get(job['arm']) or REPLAY_ARMS[job['arm']]
    cand = next(c for c in CANDIDATES if c['id'] == candidate)
    common = spec['common']
    return dict(
        protocol=spec['protocol'], protocol_sha256=sha256(PROTOCOL) if PROTOCOL.exists() else 'unfrozen',
        stage=job['kind'], dataset='canine', arm=job['arm'], model=arm['name'], seed=job['seed'], fold=1,
        candidate=candidate, lr=cand['lr'], weight_decay=cand['weight_decay'],
        epochs={'screen': 8, 'final': 30}[job['kind']], patience={'screen': None, 'final': 10}[job['kind']],
        warmup_epochs=common['warmup_epochs'], batch_size=common['batch_size'], num_workers=common['num_workers'],
        img_size=224, ema_decay=common['ema_decay'], gradient_clip=common['gradient_clip'], precision='float32',
        stochastic_depth=0. if arm['name'] == 'shvit_s1' else .1, pretrained=bool(arm.get('pretrained', False)),
        num_classes=len(classes), classes=classes, split_sha256=split_sha,
        source_code_sha256=spec['source_code_sha256'], test_used_for_selection=False)


def make_model(cfg):
    if cfg['model'] == 'gfnet_matched':
        return build_model('gfnet_matched', classes=cfg['num_classes'], size=cfg['img_size'])
    if cfg['model'] not in {a['name'] for a in ARMS.values()}:
        raise ValueError(cfg['model'])
    return build_model(cfg['model'], classes=cfg['num_classes'], size=cfg['img_size'], pretrained=cfg['pretrained'])


def train(job, device_name, out=OUT, limit=None, epochs=None, workers=None):
    smoke = limit is not None or epochs is not None
    spec = protocol_spec() if smoke else load_protocol()
    candidate = resolved_candidate(job, out)
    if candidate is None:
        raise ValueError('Job has no validation selection yet')
    directory = run_dir(job, candidate, out)
    directory.mkdir(parents=True, exist_ok=True)
    summary_path, config_path = directory / 'summary.json', directory / 'config.json'
    if summary_path.exists():
        result = json.loads(summary_path.read_text())
        if sha256(ROOT / result['checkpoint']) != result['checkpoint_sha256']:
            raise ValueError('Completed round-5 checkpoint changed')
        print('Reusing completed round-5 run', result['tag'], flush=True)
        return result
    splits = ['train', 'val'] if job['kind'] == 'screen' else ['train', 'val', 'test']
    frames, datasets, classes, split_sha = canine_data(splits, limit)
    cfg = run_config(spec, job, candidate, classes, split_sha)
    if epochs:
        cfg['epochs'] = epochs
    if config_path.exists() and json.loads(config_path.read_text()) != cfg:
        raise ValueError('Round-5 configuration changed for ' + directory.name)
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
            raise ValueError('Round-5 resume fingerprint changed')
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
    result = dict(tag=directory.name, job=job, arm=job['arm'], model=cfg['model'], dataset='canine',
                  seed=cfg['seed'], fold=cfg['fold'], candidate=candidate, stage=job['kind'],
                  best_validation_epoch=best_epoch, best_validation_macro_f1=best, epochs_completed=rows[-1]['epoch'],
                  max_epochs=cfg['epochs'], training_time_sec=total,
                  peak_gpu_allocated_mb=max(r['peak_gpu_allocated_mb'] for r in rows),
                  checkpoint=str(best_path.relative_to(ROOT) if best_path.is_relative_to(ROOT) else best_path),
                  checkpoint_sha256=sha256(best_path), config_fingerprint=fingerprint, split_sha256=cfg['split_sha256'],
                  protocol_sha256=cfg['protocol_sha256'], parameters=sum(p.numel() for p in model.parameters()),
                  trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                  pretrained=cfg['pretrained'], device=device_name, **metrics)
    write_json(summary_path, result)
    if not smoke:
        last_path.unlink()
    print('COMPLETED', directory.name, 'validation', best, flush=True)
    return result


def replay(device, scratch):
    """Re-run the first epoch of earlier runs with this code and compare with their logs."""
    rows = []
    for item in REPLAY:
        job = {k: item[k] for k in ['kind', 'arm', 'candidate', 'seed']}
        result = train(job, device, out=Path(scratch), epochs=1)
        checkpoint = Path(result['checkpoint'])
        new = pd.read_csv((checkpoint if checkpoint.is_absolute() else ROOT / checkpoint).parent / 'epochs.csv').iloc[0]
        old = pd.read_csv(ROOT / item['reference'] / 'epochs.csv').iloc[0]
        rows.append(dict(reference=item['reference'], train_loss_new=float(new.train_loss), train_loss_old=float(old.train_loss),
                         val_f1_new=float(new.validation_macro_f1), val_f1_old=float(old.validation_macro_f1),
                         agree_6dp=bool(round(new.train_loss, 6) == round(old.train_loss, 6) and
                                        round(new.validation_macro_f1, 6) == round(old.validation_macro_f1, 6))))
        print('REPLAY', json.dumps(rows[-1]), flush=True)
    write_json(OUT / 'replay_first_epoch.json', dict(rows=rows, device=device, note='epoch-1 replay with round-5 code'))
    if not all(r['agree_6dp'] for r in rows):
        raise SystemExit('Round-5 replay did not reproduce the earlier first epochs')


# ----------------------------------------------------------------------------- scheduling

def select_ready(spec, out=OUT):
    for arm in ARMS:
        target = selection_path(arm, out)
        if target.exists():
            continue
        paths = [run_dir(dict(kind='screen', arm=arm, candidate=c['id'], seed=42), None, out) / 'summary.json'
                 for c in CANDIDATES]
        if not all(p.exists() for p in paths):
            continue
        trials = [json.loads(p.read_text()) for p in paths]
        for trial, path in zip(trials, paths):
            if trial['stage'] != 'screen' or 'test' in trial or any(path.parent.glob('test*')):
                raise ValueError('Selection received a test result')
            if trial['protocol_sha256'] != sha256(PROTOCOL):
                raise ValueError('Selection protocol mismatch')
        winner = max(range(len(trials)), key=lambda i: (trials[i]['best_validation_macro_f1'], -i))
        write_json(target, dict(dataset='canine', arm=arm, candidate=CANDIDATES[winner],
                                validation_scores=[t['best_validation_macro_f1'] for t in trials],
                                screening_tags=[t['tag'] for t in trials], objective=spec['screening']['objective'],
                                protocol_sha256=sha256(PROTOCOL), test_used=False))
        print('VALIDATION SELECTED', arm, CANDIDATES[winner]['id'], flush=True)


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
            if all(state in ('done', 'failed') for _, state, _ in states):
                print('All round-5 jobs finished on', device, flush=True)
                return
            time.sleep(60)
            continue
        job, tag = nxt
        print(time.strftime('%H:%M:%S'), device, 'START', tag, flush=True)
        with (logs / f'{tag}.log').open('a') as log:
            code = subprocess.run([sys.executable, '-m', 'experiments.round5', 'job', '--spec', json.dumps(job),
                                   '--device', device], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
        if code != 0:
            write_json(OUT / 'failed' / f'{tag}.json', dict(job=job, device=device, returncode=code,
                                                         log=str((logs / f'{tag}.log').relative_to(ROOT))))
        (OUT / 'locks' / f'{tag}.lock').unlink(missing_ok=True)
        print(time.strftime('%H:%M:%S'), device, 'END', tag, 'code', code, flush=True)


def report():
    spec = load_protocol()
    rows = []
    seen = set()
    for job in job_list(spec):
        state, tag = job_state(job)
        if tag in seen:
            continue
        seen.add(tag)
        rows.append(dict(state=state, tag=tag or f"canine_{job['arm']}_<selection>_s{job['seed']}_{job['kind']}"))
    frame = pd.DataFrame(rows)
    print(frame.state.value_counts().to_string())
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
        replay(args.device, os.environ['ROUND5_SCRATCH'])
    elif args.command == 'smoke':
        scratch = Path(os.environ['ROUND5_SCRATCH'])
        for arm in ARMS:
            result = train(dict(kind='final', arm=arm, candidate='c2', seed=42), args.device, out=scratch, limit=256, epochs=1)
            print('SMOKE OK', result['tag'], result['parameters'], result['pretrained'], result['test']['macro_f1'], flush=True)


if __name__ == '__main__':
    main()
