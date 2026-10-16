"""Equal-budget validation-only screening, followed by a separate final test.

Screening never creates a test Dataset/DataLoader and never reads test predictions.
This follow-up is not a blinded new holdout: earlier shared-preset test results exist.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, WeightedRandomSampler

from .common import (CLASSES, ROOT, ManifestDataset, build_model, infer, load_manifest,
                     prediction_csv, seed_all, sha256, status, write_json)
from .training import ModelEMA, build_param_groups, seed_worker, train_one_epoch
from evaluation_metrics import compute_metric_bundle

PROTOCOL = ROOT / 'configs/validation_tuning.json'
PRETRAINED = {'deit', 'convnext', 'fastvit_t8', 'swin', 'shvit_s1'}
NO_DECAY = ('norm', 'filter_real', 'filter_imag', 'mu', 'log_sigma', 'rho_raw',
            'theta', 'log_amp', 'phi', 'w_real', 'w_imag')


def factory(name, pretrained):
    if name == 'shvit_s1':
        import timm
        # SHViT has no drop_path_rate constructor argument in this timm release.
        return timm.create_model('shvit_s1', pretrained=pretrained, num_classes=len(CLASSES))
    return build_model(name, classes=len(CLASSES), pretrained=pretrained)


def freeze_protocol():
    files = ['experiments/validation_tuning.py', 'experiments/training.py',
             'experiments/common.py', 'models/psan.py', 'models/gfnet.py',
             'models/afno.py', 'train/train_base.py']
    protocol = dict(
        protocol='equal_budget_validation_only_followup_v1',
        frozen_before_candidate_outcomes=True,
        models=['psan', 'gfnet', 'afno', 'deit', 'convnext', 'fastvit_t8', 'swin', 'shvit_s1'],
        candidates=[dict(id='c0', lr=1e-4, weight_decay=1e-3),
                    dict(id='c1', lr=3e-4, weight_decay=1e-3),
                    dict(id='c2', lr=5e-4, weight_decay=5e-2)],
        seed=42, screening_epochs=8, final_epochs=30, warmup_epochs=3,
        batch_size=128, num_workers=6, img_size=224, ema_decay=.999,
        final_patience=10, gradient_clip=1., precision='float32',
        objective='highest best-epoch validation Macro F1; ties use earlier candidate order',
        screening_early_stopping=False,
        final_default_reuse='reuse a hash-verified completed identical shared-preset run if c0 wins',
        selection_inputs=['training images and labels', 'validation images and labels'],
        forbidden_selection_inputs=['test images', 'test labels', 'test predictions or metrics'],
        split_sha256=sha256(ROOT / 'results/revision_split.csv'),
        source_code_sha256={p: sha256(ROOT / p) for p in files},
        stochastic_depth='0.1 for supported models; SHViT constructor has no stochastic-depth option',
        scope='same candidate/epoch/update budget, not a proof of optimal tuning or equal wall time',
        prior_holdout_exposure='shared-preset test results already exist; no new blinded holdout claim')
    if PROTOCOL.exists():
        old = json.loads(PROTOCOL.read_text())
        if old != protocol:
            raise ValueError('Frozen tuning protocol/source differs; preserve it and create a new protocol')
    else:
        write_json(PROTOCOL, protocol)
    return protocol


def trial_config(protocol, name, candidate, stage, device):
    return dict(
        revision_protocol=protocol['protocol'], model=name, dataset='canine', seed=protocol['seed'],
        device=device, num_classes=len(CLASSES), classes=CLASSES,
        img_size=protocol['img_size'], batch_size=protocol['batch_size'],
        num_workers=protocol['num_workers'], epochs=protocol[stage + '_epochs'],
        warmup_epochs=protocol['warmup_epochs'], ema_decay=protocol['ema_decay'],
        patience=protocol['final_patience'] if stage == 'final' else None,
        lr=candidate['lr'], weight_decay=candidate['weight_decay'], candidate=candidate['id'],
        stage=stage, pretrained=name in PRETRAINED, atom='gabor', atoms=16,
        atom_dropout=.2, init='morlet', axis_aligned=False, fixed_phase=False,
        gradient_clip=1., precision='float32', protocol_sha256=sha256(PROTOCOL),
        split_sha256=protocol['split_sha256'], source_code_sha256=protocol['source_code_sha256'],
        stochastic_depth=0. if name == 'shvit_s1' else .1,
        test_used_for_selection=False)


def train_trial(protocol, name, candidate, stage, device_name):
    cfg = trial_config(protocol, name, candidate, stage, device_name)
    tag = f'canine_{name}_{candidate["id"]}_s{protocol["seed"]}_{stage}'
    directory = ROOT / 'results/tuning' / ('trials' if stage == 'screening' else 'final') / tag
    directory.mkdir(parents=True, exist_ok=True)
    config_path = directory / 'config.json'
    summary_path = directory / 'summary.json'
    if config_path.exists() and json.loads(config_path.read_text()) != cfg:
        raise ValueError('Tuning config changed for existing ' + tag)
    if summary_path.exists():
        result = json.loads(summary_path.read_text())
        if sha256(ROOT / result['checkpoint']) != result['checkpoint_sha256']:
            raise ValueError('Tuning checkpoint changed')
        if stage == 'screening' and any(directory.glob('test*')):
            raise ValueError('Test artifact in validation-only screening directory')
        print('Reusing completed tuning run', tag, flush=True)
        return result
    write_json(config_path, cfg)
    fingerprint = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    status(tag, 'IN PROGRESS', config=str(config_path.relative_to(ROOT)))
    frame = load_manifest()
    # Test partition is not passed to any Dataset in the candidate stage.
    splits = ['train', 'val'] if stage == 'screening' else ['train', 'val', 'test']
    frames = {s: frame[frame.split == s].reset_index(drop=True) for s in splits}
    datasets = {s: ManifestDataset(f, s == 'train', cfg['img_size']) for s, f in frames.items()}
    seed_all(cfg['seed'])
    device = torch.device(device_name)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    model = factory(name, cfg['pretrained']).to(device)
    optimizer = torch.optim.AdamW(build_param_groups(model, cfg['weight_decay'], NO_DECAY), lr=cfg['lr'])
    ema = ModelEMA(model, cfg['ema_decay'], device)
    labels = datasets['train'].labels
    counts = np.bincount(labels, minlength=len(CLASSES))
    sampler_rng = torch.Generator().manual_seed(cfg['seed'] + 1000)
    loader_rng = torch.Generator().manual_seed(cfg['seed'] + 2000)
    sampler = WeightedRandomSampler(torch.as_tensor((1 / np.sqrt(counts))[labels], dtype=torch.double),
                                    len(labels), replacement=True, generator=sampler_rng)
    loaders = {s: DataLoader(d, batch_size=cfg['batch_size'],
        sampler=sampler if s == 'train' else None, shuffle=False, num_workers=cfg['num_workers'],
        pin_memory=device.type == 'cuda', worker_init_fn=seed_worker, generator=loader_rng,
        persistent_workers=False) for s, d in datasets.items()}
    start, best, best_epoch, bad, total, rows = 1, -1., 0, 0, 0., []
    last_path, best_path = directory / 'last.pth', directory / 'best.pth'
    if last_path.exists():
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        if checkpoint['config_fingerprint'] != fingerprint:
            raise ValueError('Tuning resume fingerprint changed')
        model.load_state_dict(checkpoint['model']); ema.ema.load_state_dict(checkpoint['ema'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start = checkpoint['epoch'] + 1
        best, best_epoch, bad = checkpoint['best_macro_f1'], checkpoint['best_epoch'], checkpoint['patience_counter']
        total, rows, rng = checkpoint['training_time_sec'], checkpoint['history'], checkpoint['rng']
        torch.set_rng_state(rng['torch'].cpu()); np.random.set_state(rng['numpy']); random.setstate(rng['python'])
        sampler_rng.set_state(rng['sampler'].cpu()); loader_rng.set_state(rng['loader'].cpu())
        if device.type == 'cuda':
            torch.cuda.set_rng_state(rng['cuda'].cpu(), device)
        print('Resumed', tag, start, flush=True)
    for epoch in range(start, cfg['epochs'] + 1):
        if stage == 'final' and bad >= cfg['patience']:
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
        val_f1 = float(f1_score(val_y, val_logits.argmax(-1), labels=range(len(CLASSES)), average='macro', zero_division=0))
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
            best_macro_f1=best, best_epoch=best_epoch, patience_counter=bad, training_time_sec=total, history=rows,
            rng=dict(torch=torch.get_rng_state(), numpy=np.random.get_state(), python=random.getstate(),
                     sampler=sampler_rng.get_state(), loader=loader_rng.get_state(),
                     cuda=torch.cuda.get_rng_state(device) if device.type == 'cuda' else None))
        temporary = last_path.with_suffix('.tmp'); torch.save(checkpoint, temporary); temporary.replace(last_path)
        if improved:
            temporary = best_path.with_suffix('.tmp'); torch.save(checkpoint, temporary); temporary.replace(best_path)
        pd.DataFrame(rows).to_csv(directory / 'epochs.csv', index=False)
        print(f'{tag} epoch={epoch}/{cfg["epochs"]} val_F1={val_f1:.5f} best={best:.5f} time={elapsed:.1f}s', flush=True)
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['eval_model']); model.eval()
    metrics = {}
    for split in splits[1:]:
        logits, y = infer(model, loaders[split], device)
        prediction_csv(directory / (split + '_predictions.csv'), frames[split], logits, y, CLASSES)
        bundle = compute_metric_bundle(y, torch.as_tensor(logits).softmax(-1).numpy(), CLASSES)
        write_json(directory / (split + '_metrics.json'), bundle)
        metrics[split] = bundle['metrics']
    result = dict(tag=tag, model=name, seed=cfg['seed'], dataset='canine', candidate=candidate['id'], stage=stage,
        pretrained=cfg['pretrained'], best_validation_epoch=best_epoch, best_validation_macro_f1=best,
        epochs_completed=rows[-1]['epoch'], training_time_sec=total, max_epochs=cfg['epochs'],
        peak_gpu_allocated_mb=max(r['peak_gpu_allocated_mb'] for r in rows),
        checkpoint=str(best_path.relative_to(ROOT)), checkpoint_sha256=sha256(best_path),
        config_fingerprint=fingerprint, split_sha256=cfg['split_sha256'], protocol_sha256=cfg['protocol_sha256'],
        parameters=sum(p.numel() for p in model.parameters()), **metrics)
    write_json(summary_path, result); status(tag, 'COMPLETED', summary=str(summary_path.relative_to(ROOT)))
    print('COMPLETED', tag, 'validation', best, 'stage', stage, flush=True)
    return result


def select(protocol, name):
    trials = []
    for candidate in protocol['candidates']:
        path = ROOT / 'results/tuning/trials' / f'canine_{name}_{candidate["id"]}_s{protocol["seed"]}_screening/summary.json'
        result = json.loads(path.read_text())
        if result['stage'] != 'screening' or 'test' in result:
            raise ValueError('Selection received a test result')
        if result['protocol_sha256'] != sha256(PROTOCOL):
            raise ValueError('Selection protocol mismatch')
        trials.append(result)
    winner = max(range(len(trials)), key=lambda i: (trials[i]['best_validation_macro_f1'], -i))
    selected = dict(model=name, candidate=protocol['candidates'][winner],
        selection_objective=protocol['objective'], validation_scores=[r['best_validation_macro_f1'] for r in trials],
        screening_tags=[r['tag'] for r in trials], protocol_sha256=sha256(PROTOCOL), test_used=False)
    write_json(ROOT / 'results/tuning/selections' / (name + '.json'), selected)
    return selected


def reuse_default(protocol, name, selected):
    if selected['candidate']['id'] != 'c0':
        return None
    extra = '_gabor_ad0p2_M16_morlet' if name == 'psan' else ''
    if name in PRETRAINED:
        extra += '_imagenet'
    path = ROOT / 'results/training' / f'canine_{name}{extra}_s42/summary.json'
    if not path.exists():
        return None
    result = json.loads(path.read_text())
    cfg = json.loads(path.with_name('config.json').read_text())
    expected = dict(lr=1e-4, weight_decay=1e-3, seed=42, epochs=30, warmup_epochs=3,
                    batch_size=128, img_size=224, ema_decay=.999, patience=10, pretrained=name in PRETRAINED,
                    split_sha256=protocol['split_sha256'])
    if any(cfg.get(k) != v for k, v in expected.items()):
        raise ValueError('Completed default protocol differs from final selected protocol')
    if sha256(ROOT / result['checkpoint']) != result['checkpoint_sha256']:
        raise ValueError('Completed default checkpoint differs')
    return dict(result, stage='final', candidate='c0', reused_completed_run=True,
                reused_summary=str(path.relative_to(ROOT)), protocol_sha256=sha256(PROTOCOL),
                selection_file=str((ROOT / 'results/tuning/selections' / (name + '.json')).relative_to(ROOT)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--worker', type=int, choices=[0, 1])
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--screen-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    protocol = freeze_protocol()
    if args.freeze:
        print('Frozen validation-only protocol', sha256(PROTOCOL), flush=True)
        return
    if args.worker is None:
        parser.error('--worker is required unless --freeze is used')
    names = protocol['models'][args.worker::2]
    for name in names:
        try:
            for candidate in protocol['candidates']:
                train_trial(protocol, name, candidate, 'screening', args.device)
                torch.cuda.empty_cache()
            selected = select(protocol, name)
            print('VALIDATION SELECTED', name, selected, flush=True)
            if not args.screen_only:
                final = reuse_default(protocol, name, selected)
                if final is None:
                    final = train_trial(protocol, name, selected['candidate'], 'final', args.device)
                write_json(ROOT / 'results/tuning/final_selected' / (name + '.json'), final)
                torch.cuda.empty_cache()
        except Exception:
            error = traceback.format_exc()
            status('tuning_' + name, 'BLOCKED', error=error)
            print(error, flush=True)
            raise
    status('tuning_worker' + str(args.worker), 'COMPLETED', models=names,
           screening_only=args.screen_only)


if __name__ == '__main__':
    main()
