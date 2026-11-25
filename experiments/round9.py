"""Post hoc pre-upload controls (frozen 13 November 2026, after an internal review of the revised manuscript).

wd     Matched dense control whose spectral filter receives the weight decay of its training setting.
       Canine corpus, validation-selected protocol of the earlier arms: the three settings are screened for
       eight epochs at seed 42 on training and validation data, and the selected setting is trained for at
       most 30 epochs (patience 10) at seeds 42-44. Only the no-decay list differs from the matched dense control.
steps  BloodMNIST128 and corrected DermaMNIST128 at full support with the fixed c2 recipe of the round-7
       support study, but with an epoch count that matches the optimizer steps of a 30-epoch canine run
       (PSAN AD 0, matched dense, free rank-16; seeds 42-44; warm-up scaled; no early stopping). The 30-epoch
       cells of the support study are the short-schedule counterparts.

    $PY -m experiments.round9 replay --device cuda:0      # first-epoch reproduction of three archived runs
    $PY -m experiments.round9 freeze
    $PY -m experiments.round9 worker --exp wd-screen --device cuda:0 --shard 0 --shards 3
    $PY -m experiments.round9 select
    $PY -m experiments.round9 worker --exp wd-final --device cuda:0 --shard 0 --shards 3
    $PY -m experiments.round9 worker --exp steps --device cuda:0 --shard 0 --shards 3
    $PY -m experiments.round9 report
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import shutil
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from .common import (CLASSES, ROOT, ManifestDataset, build_model, infer, load_manifest, prediction_csv, seed_all,
                     sha256, write_json)
from .round3 import external_data
from .round6 import CANDIDATES, NO_DECAY
from .round7_support import choose_support, make_model as support_model
from .training import ModelEMA, build_param_groups, seed_worker, train_one_epoch
from evaluation_metrics import compute_metric_bundle

OUT = ROOT / 'results/round9'
PROTOCOL = ROOT / 'configs/round9_protocol.json'
REPLAY = OUT / 'checks/replay.json'
SEEDS = [42, 43, 44]
WD_ARM = 'gfnet_matched_wd'
WD_NO_DECAY = tuple(name for name in NO_DECAY if name not in ('filter_real', 'filter_imag'))
STEP_DATASETS = ['bloodmnist128', 'dermamnist_c128']
STEP_ARMS = ['psan_ad0', 'gfnet_matched', 'lowrank16']
BATCH = 128
CANINE_TRAIN = 62422
CANINE_STEPS = 30 * math.ceil(CANINE_TRAIN / BATCH)        # optimizer steps of a 30-epoch canine run
CANINE_WARMUP_STEPS = 3 * math.ceil(CANINE_TRAIN / BATCH)
SOURCES = ['experiments/round9.py', 'experiments/round9_analysis.py', 'experiments/common.py', 'experiments/round3.py',
           'experiments/round6.py', 'experiments/round7_support.py', 'experiments/training.py', 'models/psan.py',
           'models/gfnet.py', 'models/lowrank.py', 'train/train_base.py', 'evaluation_metrics.py']
# Archived runs whose first epoch this code must reproduce exactly before the protocol is frozen.
REPLAYS = [dict(exp='wd', arm='gfnet_matched', candidate='c2', seed=42,
                archived='results/round3/tuning/final/canine_gfnet_matched_c2_s42_final'),
           dict(exp='steps', dataset='bloodmnist128', arm='psan_ad0', seed=42, epochs=30, warmup=3,
                archived='results/round7/support/final/bloodmnist128_psan_ad0_f100_c2_s42_final'),
           dict(exp='steps', dataset='dermamnist_c128', arm='gfnet_matched', seed=43, epochs=30, warmup=3,
                archived='results/round7/support/final/dermamnist_c128_gfnet_matched_f100_c2_s43_final')]


def step_schedule(training_images):
    """Epochs and warm-up epochs whose optimizer steps match the 30-epoch canine runs (round half up)."""
    per_epoch = math.ceil(training_images / BATCH)
    return dict(steps_per_epoch=per_epoch, epochs=math.floor(CANINE_STEPS / per_epoch + .5),
                warmup_epochs=math.floor(CANINE_WARMUP_STEPS / per_epoch + .5))


def jobs(exp):
    if exp == 'wd-screen':
        return [dict(exp='wd', kind='screen', arm=WD_ARM, candidate=c['id'], seed=42) for c in CANDIDATES]
    if exp == 'wd-final':
        return [dict(exp='wd', kind='final', arm=WD_ARM, candidate=None, seed=s) for s in SEEDS]
    if exp == 'steps':
        return [dict(exp='steps', kind='final', dataset=d, arm=a, candidate='c2', seed=s)
                for d in STEP_DATASETS for s in SEEDS for a in STEP_ARMS]
    raise ValueError(exp)


def selection_path(out=OUT):
    return out / 'wd/selection.json'


def decay_split(model, no_decay):
    """Trainable parameter counts in the decay and no-decay groups (rule of train_base.build_param_groups)."""
    decay = exempt = 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or name.endswith('.bias') or any(k in name for k in no_decay):
            exempt += p.numel()
        else:
            decay += p.numel()
    return dict(decay=decay, no_decay=exempt)


def protocol_spec():
    schedules = {}
    for dataset in STEP_DATASETS:
        frames, _, _, split_sha, _ = external_data(dataset, ['train'])
        schedules[dataset] = dict(training_images=len(frames['train']), split_sha256=split_sha,
                                  **step_schedule(len(frames['train'])))
        schedules[dataset]['optimizer_steps'] = schedules[dataset]['epochs'] * schedules[dataset]['steps_per_epoch']
    dense = build_model('gfnet_matched', classes=len(CLASSES), size=224)
    return dict(
        protocol='round9_filter_weight_decay_and_step_matched_external_v1', frozen_on='2026-11-13', post_hoc=True,
        not_a_blinded_holdout=True,
        prior_exposure=['All submitted and reviewer-driven canine and MedMNIST+ test results were known.',
                        'An internal pre-upload review had named unexplored dense-filter weight decay and the short '
                        'optimizer-step budget of the two small external datasets as open explanations; the revised '
                        'text stating both as untested was written before these runs.'],
        questions=dict(
            wd='Does the validation-selected PSAN advantage over the matched dense control persist when the dense '
               'filter receives the weight decay of its training setting?',
            steps='Does the PSAN-control contrast on BloodMNIST and corrected DermaMNIST change when the fixed c2 '
                  'recipe is trained for as many optimizer steps as a 30-epoch canine run?'),
        common=dict(batch_size=BATCH, ema_decay=.999, gradient_clip=1., stochastic_depth=.1,
                    precision='float32; CUDA matmul TF32 off', sampler='inverse-square-root class weights, replacement, '
                    'one training-set size of draws per epoch', checkpoint='best EMA validation Macro F1',
                    test='evaluated once after validation selection; never used for selection'),
        wd=dict(dataset='canine (main audited split)', arm=WD_ARM, model='gfnet_matched', seeds=SEEDS,
                candidates=CANDIDATES, screening='8 epochs at seed 42, no early stopping, training/validation data only',
                selection='highest best validation Macro F1; ties to the earlier candidate',
                final=dict(epochs=30, patience=10, warmup_epochs=3, num_workers=6),
                no_weight_decay=list(WD_NO_DECAY), control_no_weight_decay=list(NO_DECAY),
                parameter_groups=dict(control=decay_split(dense, NO_DECAY), with_filter_decay=decay_split(dense, WD_NO_DECAY)),
                difference='filter_real and filter_imag of the twelve dense filters move from the no-decay group to the '
                           'decay group; architecture, initialization, data, sampler, schedule and selection rule are unchanged',
                comparators=dict(dense='results/round3/tuning/final/canine_gfnet_matched_c2_s{seed}_final',
                                 psan=['results/tuning/final/canine_psan_c2_s42_final',
                                       'results/round3/tuning/final/canine_psan_c2_s43_final',
                                       'results/round3/tuning/final/canine_psan_c2_s44_final'],
                                 psan_ad0='results/round3/tuning/final/canine_psan_ad0_c2_s{seed}_final')),
        steps=dict(datasets=STEP_DATASETS, arms=STEP_ARMS, seeds=SEEDS, support_percentage=100, candidate='c2',
                   lr=5e-4, weight_decay=.05, num_workers=4, early_stopping=False, cosine_over_full_budget=True,
                   canine_reference=dict(training_images=CANINE_TRAIN, optimizer_steps=CANINE_STEPS,
                                         warmup_steps=CANINE_WARMUP_STEPS),
                   schedules=schedules, no_weight_decay=list(NO_DECAY),
                   short_schedule_counterparts='results/round7/support/final/{dataset}_{arm}_f100_c2_s{seed}_final '
                                               '(30 epochs, 3 warm-up epochs; otherwise the same recipe, data order and initialization)'),
        statistics=dict(bootstrap=10000, seed=20261113,
                        resampling='crossed bootstrap of test groups (visual groups for canine, lesions for corrected '
                                   'Derma, images for Blood) and the three seed indices; 95% percentile intervals, unadjusted',
                        families=dict(
                            wd='3 Holm-adjusted contrasts: dense with filter decay minus dense; PSAN (AD 0.2) minus dense '
                               'with filter decay; PSAN (AD 0) minus dense with filter decay',
                            steps_simple='per dataset, 2 Holm-adjusted contrasts at the step-matched schedule: PSAN (AD 0) '
                                         'minus matched dense and minus free rank-16',
                            steps_interaction='per dataset, 2 Holm-adjusted contrasts: (PSAN minus control) at the '
                                              'step-matched schedule minus the same contrast at 30 epochs'),
                        all_runs='every screening trial, final run and contrast is reported; no outcome-based extension or exclusion'),
        replay='results/round9/checks/replay.json: first epoch of three archived runs reproduced by this code before freezing',
        source_code_sha256={p: sha256(ROOT / p) for p in SOURCES})


def freeze():
    proof = json.loads(REPLAY.read_text())
    if not proof['passed'] or proof['source_sha256'] != sha256(__file__):
        raise ValueError('The current source needs a successful replay before the protocol is frozen')
    spec = protocol_spec()
    if PROTOCOL.exists() and json.loads(PROTOCOL.read_text()) != spec:
        raise ValueError('Frozen round-9 protocol differs; create a new version')
    if not PROTOCOL.exists():
        write_json(PROTOCOL, spec)
    return spec


def load_protocol():
    spec = json.loads(PROTOCOL.read_text())
    current = {p: sha256(ROOT / p) for p in SOURCES if p != 'experiments/round9_analysis.py'}
    changed = [p for p, digest in current.items() if spec['source_code_sha256'][p] != digest]
    if changed:
        raise ValueError('Training sources changed after the freeze: ' + ', '.join(changed))
    return spec


def fit(cfg, directory, make, no_decay, datasets, frames, weights, classes, device_name, splits, stop_after=None):
    """Training loop of experiments.round6.train / round7_support.train with the schedule taken from cfg."""
    config_path, summary_path = directory / 'config.json', directory / 'summary.json'
    if config_path.exists() and json.loads(config_path.read_text()) != cfg:
        raise ValueError('Round-9 configuration changed for ' + directory.name)
    write_json(config_path, cfg)
    fingerprint = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    seed_all(cfg['seed'])
    device = torch.device(device_name)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    model = make().to(device)
    optimizer = torch.optim.AdamW(build_param_groups(model, cfg['weight_decay'], no_decay), lr=cfg['lr'])
    ema = ModelEMA(model, cfg['ema_decay'], device)
    sampler_rng = torch.Generator().manual_seed(cfg['seed'] + 1000)
    loader_rng = torch.Generator().manual_seed(cfg['seed'] + 2000)
    sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), cfg['sampler_draws_per_epoch'],
                                    replacement=True, generator=sampler_rng)
    loaders = {s: DataLoader(d, batch_size=cfg['batch_size'], sampler=sampler if s == 'train' else None,
                             shuffle=False, num_workers=cfg['num_workers'], pin_memory=device.type == 'cuda',
                             worker_init_fn=seed_worker, generator=loader_rng, persistent_workers=False)
               for s, d in datasets.items()}
    start, best, best_epoch, bad, total, rows = 1, -1., 0, 0, 0., []
    last_path, best_path = directory / 'last.pth', directory / 'best.pth'
    if last_path.exists() and stop_after is None:
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        if checkpoint['config_fingerprint'] != fingerprint:
            raise ValueError('Round-9 resume fingerprint changed')
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
    print('START', directory.name, json.dumps({k: v for k, v in cfg.items() if k not in ('classes', 'source_code_sha256')}), flush=True)
    warmup = cfg['warmup_epochs']
    for epoch in range(start, cfg['epochs'] + 1):
        if cfg['patience'] is not None and bad >= cfg['patience']:
            break
        tic = time.perf_counter()
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
        rows.append(dict(epoch=epoch, train_loss=loss, train_accuracy=accuracy / 100, validation_macro_f1=val_f1,
                         learning_rate=cfg['lr'] * factor, epoch_time_sec=elapsed, training_time_sec=total,
                         gradient_norm_mean=extra['gradient_norm_mean']))
        if stop_after is not None:
            if epoch >= stop_after:
                return rows
            continue
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
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=False)['eval_model']); model.eval()
    metrics = {}
    for split in splits[1:]:
        logits, y = infer(model, loaders[split], device)
        prediction_csv(directory / (split + '_predictions.csv'), frames[split], logits, y, classes)
        bundle = compute_metric_bundle(y, torch.as_tensor(logits).softmax(-1).numpy(), classes)
        write_json(directory / (split + '_metrics.json'), bundle)
        metrics[split] = bundle['metrics']
    result = dict(tag=directory.name, experiment=cfg['experiment'], stage=cfg['stage'], dataset=cfg['dataset'],
                  arm=cfg['arm'], seed=cfg['seed'], candidate=cfg['candidate'], best_validation_epoch=best_epoch,
                  best_validation_macro_f1=best, epochs_completed=rows[-1]['epoch'], max_epochs=cfg['epochs'],
                  optimizer_steps_completed=rows[-1]['epoch'] * math.ceil(cfg['sampler_draws_per_epoch'] / cfg['batch_size']),
                  training_time_sec=total, checkpoint=str(best_path.relative_to(ROOT)), checkpoint_sha256=sha256(best_path),
                  config_fingerprint=fingerprint, protocol_sha256=cfg['protocol_sha256'], split_sha256=cfg['split_sha256'],
                  device=device_name, **metrics)
    write_json(summary_path, result)
    last_path.unlink(missing_ok=True)
    del model, ema, optimizer, loaders
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    print('COMPLETED', directory.name, 'validation', best, flush=True)
    return result


def completed(directory):
    path = directory / 'summary.json'
    if not path.exists():
        return None
    result = json.loads(path.read_text())
    if sha256(ROOT / result['checkpoint']) != result['checkpoint_sha256']:
        raise ValueError('Completed round-9 checkpoint changed: ' + directory.name)
    return result


def train_wd(job, device_name, spec, out=OUT, stop_after=None):
    arm = job['arm']
    candidate = job['candidate']
    if candidate is None:
        candidate = json.loads(selection_path(out).read_text())['selected']
    cand = next(c for c in CANDIDATES if c['id'] == candidate)
    screen = job['kind'] == 'screen'
    directory = out / 'wd' / ('trials' if screen else 'final') / \
        f'canine_{arm}_{candidate}_s{job["seed"]}_{"screening" if screen else "final"}'
    directory.mkdir(parents=True, exist_ok=True)
    if stop_after is None and completed(directory):
        return completed(directory)
    splits = ['train', 'val'] if screen or stop_after else ['train', 'val', 'test']
    frame = load_manifest()
    frames = {s: frame[frame.split == s].reset_index(drop=True) for s in splits}
    datasets = {s: ManifestDataset(f, s == 'train', 224) for s, f in frames.items()}
    if len(frames['train']) != CANINE_TRAIN:
        raise ValueError('Unexpected canine training-set size')
    labels = datasets['train'].labels
    counts = np.bincount(labels, minlength=len(CLASSES))
    no_decay = WD_NO_DECAY if arm == WD_ARM else NO_DECAY
    cfg = dict(protocol=spec['protocol'], protocol_sha256=sha256(PROTOCOL) if PROTOCOL.exists() else 'unfrozen',
               experiment='wd', stage=job['kind'], dataset='canine', arm=arm, model='gfnet_matched', seed=job['seed'],
               candidate=candidate, lr=cand['lr'], weight_decay=cand['weight_decay'], epochs=8 if screen else 30,
               patience=None if screen else 10, warmup_epochs=3, batch_size=BATCH, num_workers=6, img_size=224,
               ema_decay=.999, gradient_clip=1., no_weight_decay=list(no_decay),
               filter_weight_decay=arm == WD_ARM, sampler_draws_per_epoch=len(labels), num_classes=len(CLASSES),
               classes=CLASSES, split_sha256=sha256(ROOT / 'results/revision_split.csv'),
               source_code_sha256=spec['source_code_sha256'], test_used_for_selection=False)
    make = lambda: build_model('gfnet_matched', classes=len(CLASSES), size=224)
    return fit(cfg, directory, make, no_decay, datasets, frames, (1 / np.sqrt(counts))[labels], CLASSES,
               device_name, splits, stop_after)


def train_steps(job, device_name, spec, out=OUT, schedule=None, stop_after=None):
    dataset, arm, seed = job['dataset'], job['arm'], job['seed']
    splits = ['train', 'val'] if stop_after else ['train', 'val', 'test']
    frames, datasets, classes, split_sha, _ = external_data(dataset, splits)
    schedule = schedule or step_schedule(len(frames['train']))
    directory = out / 'steps/final' / f'{dataset}_{arm}_e{schedule["epochs"]}_c2_s{seed}_final'
    directory.mkdir(parents=True, exist_ok=True)
    if stop_after is None and completed(directory):
        return completed(directory)
    full_n = len(frames['train'])
    full_counts = np.bincount(frames['train'].label, minlength=len(classes))
    ids = choose_support(frames['train'], 100, seed)          # full support: every training image, as in the f100 cells
    if len(ids) != full_n:
        raise ValueError('Full support must retain every training image')
    datasets['train'] = Subset(datasets['train'], ids.tolist())
    labels = frames['train'].label.to_numpy()[ids]
    subset_counts = np.bincount(labels, minlength=len(classes))
    weights = np.sqrt(full_counts)[labels] / subset_counts[labels]
    cfg = dict(protocol=spec['protocol'], protocol_sha256=sha256(PROTOCOL) if PROTOCOL.exists() else 'unfrozen',
               experiment='steps', stage='final', dataset=dataset, arm=arm,
               model='lowrank' if arm == 'lowrank16' else 'gfnet_matched' if arm == 'gfnet_matched' else 'psan',
               seed=seed, candidate='c2', lr=5e-4, weight_decay=.05, epochs=schedule['epochs'], patience=None,
               warmup_epochs=schedule['warmup_epochs'], batch_size=BATCH, num_workers=4, img_size=128,
               ema_decay=.999, gradient_clip=1., atoms=16, atom_dropout=0., no_weight_decay=list(NO_DECAY),
               filter_weight_decay=False, support_percentage=100, sampler_draws_per_epoch=full_n,
               num_classes=len(classes), classes=classes, split_sha256=split_sha,
               source_code_sha256=spec['source_code_sha256'], test_used_for_selection=False)
    make = lambda: support_model(arm, len(classes))
    return fit(cfg, directory, make, NO_DECAY, datasets, frames, weights, classes, device_name, splits, stop_after)


def replay(device):
    """Repeat the first epoch of three archived runs with this code; nothing here uses test data."""
    scratch = OUT / 'checks/replay_scratch'
    if scratch.exists():
        shutil.rmtree(scratch)
    spec = protocol_spec()
    rows = []
    for item in REPLAYS:
        archived = pd.read_csv(ROOT / item['archived'] / 'epochs.csv').iloc[0]
        if item['exp'] == 'wd':
            job = dict(exp='wd', kind='final', arm=item['arm'], candidate=item['candidate'], seed=item['seed'])
            new = train_wd(job, device, spec, out=scratch, stop_after=1)[0]
        else:
            job = dict(exp='steps', kind='final', dataset=item['dataset'], arm=item['arm'], candidate='c2', seed=item['seed'])
            new = train_steps(job, device, spec, out=scratch, stop_after=1,
                              schedule=dict(epochs=item['epochs'], warmup_epochs=item['warmup']))[0]
        row = dict(archived_run=item['archived'],
                   archived_train_loss=float(archived.train_loss), replay_train_loss=float(new['train_loss']),
                   archived_validation_macro_f1=float(archived.validation_macro_f1),
                   replay_validation_macro_f1=float(new['validation_macro_f1']))
        row['exact'] = bool(abs(row['archived_train_loss'] - row['replay_train_loss']) < 5e-7 and
                            abs(row['archived_validation_macro_f1'] - row['replay_validation_macro_f1']) < 5e-7)
        rows.append(row)
        print(json.dumps(row), flush=True)
    shutil.rmtree(scratch)
    write_json(REPLAY, dict(passed=all(r['exact'] for r in rows), device=device, runs=rows, source_sha256=sha256(__file__),
                            note='First epoch only; schedules of the archived runs; the dense replay uses the original '
                                 'no-decay list, so the weight-decay arm differs from a reproduced path by that list alone.'))
    if not all(r['exact'] for r in rows):
        raise SystemExit('Replay did not reproduce the archived first epochs')


def select(out=OUT):
    trials = []
    for job in jobs('wd-screen'):
        result = completed(out / 'wd/trials' / f'canine_{job["arm"]}_{job["candidate"]}_s42_screening')
        if result is None or result['epochs_completed'] != 8:
            raise ValueError('Screening is incomplete: ' + job['candidate'])
        trials.append(dict(candidate=job['candidate'], best_validation_macro_f1=result['best_validation_macro_f1'],
                           best_validation_epoch=result['best_validation_epoch']))
    winner = max(range(len(trials)), key=lambda i: (trials[i]['best_validation_macro_f1'], -i))
    record = dict(arm=WD_ARM, selected=trials[winner]['candidate'], trials=trials,
                  rule='highest best validation Macro F1 within eight epochs; ties to the earlier candidate; no test data')
    path = selection_path(out)
    if path.exists() and json.loads(path.read_text()) != record:
        raise ValueError('Existing selection differs')
    write_json(path, record)
    print(json.dumps(record), flush=True)
    return record


def worker(exp, device, shard, shards):
    spec = load_protocol()
    for job in jobs(exp)[shard::shards]:
        (train_wd if job['exp'] == 'wd' else train_steps)(job, device, spec)


def report():
    rows = []
    for path in sorted(OUT.glob('*/*/*/summary.json')):
        r = json.loads(path.read_text())
        rows.append(dict(tag=r['tag'], best_epoch=r['best_validation_epoch'], epochs=r['epochs_completed'],
                         val=round(r['best_validation_macro_f1'], 4),
                         test=round(r['test']['macro_f1'], 4) if 'test' in r else None,
                         minutes=round(r['training_time_sec'] / 60, 1)))
    print(pd.DataFrame(rows).to_string(index=False) if rows else 'no completed runs', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['replay', 'freeze', 'worker', 'select', 'report'])
    parser.add_argument('--exp', choices=['wd-screen', 'wd-final', 'steps'])
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    args = parser.parse_args()
    if args.action == 'replay':
        replay(args.device)
    elif args.action == 'freeze':
        freeze(); print('Frozen round-9 protocol.', flush=True)
    elif args.action == 'select':
        select()
    elif args.action == 'report':
        report()
    else:
        worker(args.exp, args.device, args.shard, args.shards)
