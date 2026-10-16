from __future__ import annotations

import hashlib
import json
import os
import random
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
CLASSES = sorted(p.name for p in (ROOT / 'data_raw').iterdir() if p.is_dir())
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def write_json(path, obj):
    # Empty calibration bins/undefined subgroup AUC are genuinely undefined.
    # Represent them as null, never fabricate a number or emit nonstandard NaN.
    def clean(value):
        if isinstance(value, dict): return {str(k):clean(v) for k,v in value.items()}
        if isinstance(value, (list,tuple)): return [clean(v) for v in value]
        if isinstance(value, np.ndarray): return clean(value.tolist())
        if isinstance(value, (float,np.floating)):
            return float(value) if np.isfinite(value) else None
        if isinstance(value, np.integer): return int(value)
        if isinstance(value, np.bool_): return bool(value)
        return value
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Concurrent runs can update the same phase status; use a unique atomic writer.
    tmp = path.with_suffix(path.suffix + f'.{os.getpid()}.{uuid.uuid4().hex}.tmp')
    tmp.write_text(json.dumps(clean(obj), indent=2, ensure_ascii=False, default=str, allow_nan=False) + '\n')
    os.replace(tmp, path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False


def transform(train=False, size=224):
    items = [transforms.Resize((size, size))]
    if train:
        items += [transforms.RandomHorizontalFlip(), transforms.ColorJitter(.15, .15, .10, .05)]
    return transforms.Compose(items + [transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


class ManifestDataset(Dataset):
    def __init__(self, frame, train=False, size=224):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform(train, size)
        self.classes = CLASSES
        self.labels = self.frame.label.to_numpy(dtype=np.int64)
        cache_dir = ROOT / 'results' / 'cache'
        self.cached = None
        self.cache_indices = None
        if size == 224 and (cache_dir/'canine_224_complete.json').exists():
            provenance = json.loads((cache_dir/'canine_224_complete.json').read_text())
            if provenance['frozen_manifest_sha256'] != sha256(ROOT/'cv_splits/stratified_group_5fold.csv'):
                raise ValueError('Stale image cache')
            cache_paths = pd.read_csv(cache_dir/'canine_224_paths.csv').path.tolist()
            mapping = {p:i for i,p in enumerate(cache_paths)}
            # A corrected audit can choose a different duplicate representative.
            # Missing cache entries are decoded from immutable raw files.
            self.cache_indices = [mapping.get(p,-1) for p in self.frame.path]
            self.cached = np.load(cache_dir/'canine_224.npy', mmap_mode='r')

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, i):
        row = self.frame.iloc[i]
        if self.cached is not None and self.cache_indices[i] >= 0:
            x = self.transform(Image.fromarray(self.cached[self.cache_indices[i]]))
        else:
            with Image.open(ROOT / 'data_raw' / row.path) as img:
                x = self.transform(img.convert('RGB'))
        return x, int(row.label)


def load_manifest():
    path=ROOT/'results/revision_split.csv'
    proof=json.loads((ROOT/'results/corrected_integrity.json').read_text())
    if sha256(path)!=proof['revision_manifest_sha256']:
        raise ValueError('Revision split provenance mismatch')
    if proof['cross_split_candidate_pairs'] or proof['retained_verified_pairs']:
        raise ValueError('Unresolved near duplicates in active cohort')
    return pd.read_csv(path, keep_default_na=False)


def build_model(name, classes=11, size=224, atom='gabor', dropout=.2, atoms=16,
                init='morlet', axis_aligned=False, fixed_phase=False, pretrained=False):
    if name == 'psan':
        from models.psan import psan_ti
        return psan_ti(num_classes=classes, img_size=size, drop_path_rate=.1,
                       psan_kwargs=dict(M=atoms, atom=atom, atom_dropout=dropout,
                                        init=init, anisotropic=not axis_aligned,
                                        learn_phase=not fixed_phase))
    if name == 'gfnet':
        from models.gfnet import gfnet_tiny
        return gfnet_tiny(num_classes=classes, img_size=size, drop_path_rate=.1)
    if name == 'gfnet_matched':
        # Isolate the mixer: identical PSAN stem, position embedding, MLPs,
        # normalization and residual paths; substitute only the dense filter.
        from models.psan import psan_ti
        from models.gfnet import GlobalFilter
        model=psan_ti(num_classes=classes,img_size=size,drop_path_rate=.1)
        for block in model.blocks:
            cfg=block.filter.cfg
            block.filter=GlobalFilter(cfg.dim,cfg.h,cfg.w)
        return model
    if name == 'afno':
        from models.afno import afno_tiny
        return afno_tiny(num_classes=classes, img_size=size, drop_path_rate=.1)
    import timm
    names = {'deit': 'deit_tiny_patch16_224', 'convnext': 'convnext_atto',
             'swin': 'swin_tiny_patch4_window7_224'}
    return timm.create_model(names.get(name,name), pretrained=pretrained, num_classes=classes,
                             drop_path_rate=.1)


def load_model(checkpoint, device='cpu'):
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
    cfg = ckpt['config']
    if 'revision_protocol' in cfg:
        model = build_model(cfg['model'], classes=cfg['num_classes'], size=cfg['img_size'],
                            atom=cfg.get('atom','gabor'), dropout=cfg.get('atom_dropout',.2),
                            atoms=cfg.get('atoms',16), init=cfg.get('init','morlet'),
                            axis_aligned=cfg.get('axis_aligned',False),
                            fixed_phase=cfg.get('fixed_phase',False))
    else:
        from eval.eval_psan import build_model_from_ckpt_config
        model = build_model_from_ckpt_config(cfg, cfg.get('num_classes',11), cfg.get('img_size',224))
    source = next(k for k in ('eval_model','ema','model') if ckpt.get(k) is not None)
    sd = {k:v for k,v in ckpt[source].items() if not k.endswith(('.filter.U','.filter.V'))}
    model.load_state_dict(sd, strict=True)
    return model.to(device).eval(), ckpt, source


@torch.no_grad()
def infer(model, loader, device):
    model.eval()
    logits, labels = [], []
    for x, y in loader:
        logits.append(model(x.to(device, non_blocking=True)).float().cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(logits), np.concatenate(labels)


def prediction_csv(path, frame, logits, labels, classes):
    p = torch.as_tensor(logits).softmax(-1).numpy()
    out = frame.copy().reset_index(drop=True)
    if len(out) != len(labels) or not np.array_equal(out.label.to_numpy(), labels):
        raise ValueError('Prediction rows/labels do not align')
    out['pred_label'] = p.argmax(-1)
    for i,c in enumerate(classes):
        out['prob::' + c] = p[:,i]
        out['logit::' + c] = logits[:,i]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)


def status(phase, state, **kwargs):
    import datetime
    write_json(ROOT / 'results' / 'status' / (phase + '.json'),
               dict(phase=phase, status=state, updated_at=datetime.datetime.now().isoformat(), **kwargs))
