"""Independently verify the available cohort; never infer patient identities."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import chi2_contingency

from .common import ROOT, CLASSES, sha256, status, write_json


def metadata_record(path):
    obj = json.loads(path.read_text())
    meta = obj.get('images', {}).get('meta', {})
    row = {'path': str(path.with_suffix('').relative_to(ROOT / 'data_raw')),
           'class': path.parent.name, 'meta_keys': '|'.join(sorted(meta))}
    for key in ('file_name','date_time','device','gender','age','breed','medical_type','eye_position'):
        row[key] = meta.get(key)
    for key in ('dog_id','patient_id','visit_id','hospital_id','session_id'):
        row[key] = meta.get(key)
    return row


def verify_record(row):
    path = ROOT / 'data_raw' / row.path
    h, md5 = hashlib.sha256(), hashlib.md5()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk); md5.update(chunk)
    return {'path': row.path, 'sha256_actual': h.hexdigest(), 'md5': md5.hexdigest(),
            'matches_frozen_sha256': h.hexdigest() == row.sha256, 'bytes': path.stat().st_size}


def run(args):
    """Rebuild full evidence if absent; the old retained-only audit is insufficient."""
    out=ROOT/'results'
    for name in ['tables','figures','experiment_logs']:(out/name).mkdir(parents=True,exist_ok=True)
    status('audit','IN PROGRESS')
    if not (out/'raw_metadata.csv').exists():
        paths=sorted((ROOT/'data_raw').glob('*/*.json'))
        with ThreadPoolExecutor(max_workers=16) as pool:
            pd.DataFrame(pool.map(metadata_record,paths)).to_csv(out/'raw_metadata.csv',index=False)
    if not (out/'reconstructed_dedup/dedup_manifest.csv').exists():
        command=[sys.executable,str(ROOT/'deduplicate_dataset.py'),'--root',str(ROOT/'data_raw'),
                 '--out',str(out/'reconstructed_dedup'),'--workers','12','--accept-verified-near']
        with (out/'experiment_logs/raw_dedup_rebuild.log').open('w') as logfile:
            subprocess.run(command,cwd=ROOT,stdout=logfile,stderr=subprocess.STDOUT,check=True)
    from .integrity import run as corrected_run
    return corrected_run(args)


def legacy_frozen_audit(args):
    """Superseded intake audit preserved for historical traceability only."""
    status('audit','IN PROGRESS')
    out = ROOT / 'results'
    fold = pd.read_csv(ROOT / 'cv_splits/stratified_group_5fold.csv', keep_default_na=False)
    assert len(fold) == 104053 and not fold.path.duplicated().any()
    if (fold.groupby('group_id').fold.nunique() > 1).any():
        raise AssertionError('Frozen visual group crosses folds')
    summary = json.loads((ROOT / 'cv_splits/stratified_group_5fold_summary.json').read_text())
    assert sha256(ROOT / 'cv_splits/stratified_group_5fold.csv') == summary['fold_manifest_sha256']
    inventory = []
    meta_files = []
    for cls in CLASSES:
        counts = Counter()
        for path in (ROOT / 'data_raw' / cls).iterdir():
            counts[path.suffix.lower()] += 1
            if path.suffix.lower() == '.json': meta_files.append(path)
        inventory.append({'class':cls, **counts})
    pd.DataFrame(inventory).fillna(0).to_csv(out/'tables/raw_inventory.csv', index=False)
    metadata_path = out/'raw_metadata.csv'
    if not metadata_path.exists():
        with ThreadPoolExecutor(max_workers=16) as pool:
            records = list(pool.map(metadata_record, meta_files))
        pd.DataFrame(records).to_csv(metadata_path,index=False)
    meta = pd.read_csv(metadata_path, keep_default_na=False)
    verified_path = out/'retained_hash_verification.csv'
    if not verified_path.exists():
        with ThreadPoolExecutor(max_workers=12) as pool:
            rows = list(pool.map(verify_record, fold.itertuples(index=False)))
        pd.DataFrame(rows).to_csv(verified_path,index=False)
    verified = pd.read_csv(verified_path)
    assert verified.matches_frozen_sha256.all(), 'Source differs from frozen SHA-256'
    assert len(verified) == len(fold) and set(verified.path) == set(fold.path)
    assert not verified.sha256_actual.duplicated().any()
    retained = fold.assign(meta_path=fold.path.map(lambda p: str(Path(p).with_suffix(''))))
    retained = retained.merge(meta, left_on='meta_path',right_on='path',suffixes=('','_meta'),validate='one_to_one')
    assert len(retained) == len(fold)
    retained['split'] = np.where(retained.fold == 1, 'test', np.where(retained.fold == 2, 'val', 'train'))
    keep = ['path','class','label','group_id','fold','sha256','visual_group','split',
            'device','date_time','file_name','breed','gender','age','eye_position']
    retained[keep].to_csv(out/'revision_split.csv',index=False)
    for title,frame in [('raw',meta),('retained',retained)]:
        tab = pd.crosstab(frame['class'],frame.device)
        tab.to_csv(out/'tables'/f'{title}_class_device.csv')
    tab = pd.crosstab(retained['class'],retained.device)
    chi2, p, _, _ = chi2_contingency(tab)
    n = tab.to_numpy().sum()
    cramers = float(np.sqrt(chi2/n/min(tab.shape[0]-1,tab.shape[1]-1)))
    train, test = retained[retained.split=='train'],retained[retained.split=='test']
    device_prob = pd.crosstab(train.device,train.label).reindex(columns=range(len(CLASSES)),fill_value=0)
    device_prob = device_prob.div(device_prob.sum(axis=1),axis=0)
    prior = train.label.value_counts(normalize=True).reindex(range(len(CLASSES)),fill_value=0).to_numpy()
    prob = np.stack([device_prob.loc[d].to_numpy() if d in device_prob.index else prior for d in test.device])
    from evaluation_metrics import compute_metric_bundle
    bundle = compute_metric_bundle(test.label.to_numpy(),prob,CLASSES)
    write_json(out/'device_only_baseline.json',bundle)
    coverage = {key: int((meta[key] != '').sum()) for key in meta.columns if key != 'path'}
    overlap = {}
    for a,b in combinations(['train','val','test'],2):
        fa,fb = retained[retained.split==a],retained[retained.split==b]
        overlap[a+'_'+b] = {
            'paths':len(set(fa.path)&set(fb.path)),
            'visual_groups':len(set(fa.group_id)&set(fb.group_id)),
            'byte_sha256':len(set(fa.sha256)&set(fb.sha256)),
            'source_file_name':len(set(fa.file_name)&set(fb.file_name))}
    assert all(all(v == 0 for k,v in x.items() if k != 'source_file_name') for x in overlap.values())
    env = {'python':platform.python_version(),'pytorch':torch.__version__,
           'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),
           'gpu_count':torch.cuda.device_count(),
           'gpus':[{'name':torch.cuda.get_device_name(i),'memory_bytes':torch.cuda.get_device_properties(i).total_memory}
                   for i in range(torch.cuda.device_count())]}
    result = {'environment':env,'raw_images':len(meta),'retained_images':len(fold),
              'retained_groups':retained.group_id.nunique(),
              'frozen_manifest_sha256':sha256(ROOT/'cv_splits/stratified_group_5fold.csv'),
              'revision_manifest_sha256':sha256(out/'revision_split.csv'),
              'all_retained_hashes_match':bool(verified.matches_frozen_sha256.all()),
              'retained_byte_duplicate_groups':int(verified.sha256_actual.duplicated().sum()),
              'retained_md5_duplicate_rows':int(verified.md5.duplicated().sum()),
              'split_counts':retained.split.value_counts().to_dict(),
              'split_class_counts':pd.crosstab(retained['class'],retained.split).to_dict(),
              'metadata_coverage':coverage,'meta_key_sets':meta.meta_keys.value_counts().to_dict(),
              'overlap':overlap,'cramers_v':cramers,'device_chi2':float(chi2),
              'device_chi2_p':float(p),'device_only_metrics':bundle['metrics'],
              'missing_original_dedup_audit':True,
              'patient_independence_established':False,
              'dedup_exclusion_counts_independently_reverified':False}
    write_json(out/'audit.json',result)
    text = f'''# Project audit — 2026-10-12

Status: COMPLETED for available files/environment/retained byte verification; historical dedup policy verification remains BLOCKED.

## Environment and preserved evidence

Python {env['python']}; PyTorch {env['pytorch']}; CUDA {env['cuda']}; cuDNN {env['cudnn']}; {env['gpu_count']} GPUs: {env['gpus']}.
Base Python has no PyTorch; use `/home/dongbeen/anaconda3/envs/vision/bin/python`.
No Git repository exists. Source and manuscript originals, including hashes, are in `revision_history/20261012_originals/`.

All {len(meta):,} raw images have paired JSON. The frozen retained cohort contains {len(fold):,} images in {retained.group_id.nunique():,} visual groups. Every retained byte SHA-256 matches the frozen manifest; no retained SHA-256/MD5 duplicate exists. Source raw files remain untouched. Missing: original legacy split trees, clean materialization, dedup hash cache/manifest/near-pair audit, original reviewer letter, IEEE class and manuscript figures.

## Result provenance issues

The initial manuscript claims 9,122 test images and PSAN F1=0.6612/AUC=0.9657. Stored main PSAN probabilities/JSON instead contain 9,086 images, F1=0.6572525422/AUC=0.9625480165. Other legacy baselines must be matched by exact paths before paired testing. AFNO legacy predictions/checkpoint are absent. Existing multi-seed artifacts contain six baselines (42/123/2024), not repeated PSAN/AFNO.

Only CV fold1/seed42 has stored predictions; fold2 has a checkpoint but no completed metric artifact. The existing CV runner selects checkpoints on the same held-out fold it evaluates; these are validation-selected results and cannot be presented as unbiased final test performance. New scratch training uses folds3–5/train, fold2/val and fold1/test, with a fresh checkpoint and frozen configuration.

## Code findings and applied fixes

1. Width-frequency `linspace(0,pi,...)` is incorrect for odd patch grids; replaced with `2*pi*rfftfreq`. The original 14x14 grid is even and unchanged numerically. 224-resolution legacy values are not invalidated by this fix.
2. `isotropic=True` actually sets rho=0 with independently learned horizontal/vertical scales: axis-aligned, not isotropic.
3. Atom dropout rescales surviving atoms by M/survivors, not 1/(1-p). All-dropped event returns zero, with probability p^M; its unconditional expected mask weight is 1-p^M. Separate forward/decomposition calls draw independent masks in training; shared explicit masks now support exact checking.
4. Global phase is redundant with unconstrained complex mixing weights: `w*exp(i*phi)` can absorb phi. The phase ablation concerns optimization/parameterization, not added expressiveness. Gabor carrier uses norm(mu) times orientation, with the numerical 1e-4 floor; it is not a zero-mean admissible Morlet wavelet.
5. Exact atom decomposition applies to the linear filter output. Gradient scores decompose the local directional derivative; they do not sum to a nonlinear network's class logit. Spectral-CAM previously used the unfiltered spectrum. A separate `response='filtered'` variant now uses the actual real filtered output spectrum.
6. Inference cache is eval-only, non-persistent, invalidated by training, loading, device/dtype conversion, parameter version changes and explicit resolution adaptation. No change to training's dynamic filter.
7. Legacy dataset loader silently substituted black images on decode errors. It now raises an explicit path-specific error. New manifest loader also fails on decoding errors.
8. Reported profiler FLOPs omit unsupported FFT/complex/filter operations; do not call 0.744G a complete operation count. Numerical parameter counts can be independently verified.
9. Historical AFNO uses lr=5e-4/weight_decay=.05 whereas PSAN/GFNet use 1e-4/.001. No saved equal-budget tuning record exists. New runs freeze one common scratch configuration and explicitly report the absence of a search budget.

See `results/audit.json`, `results/tables/raw_inventory.csv`, and subsequent model-check JSON for machine-readable evidence.
'''
    (ROOT/'reports/00_project_audit.md').write_text(text)
    (ROOT/'reports/01_data_integrity_and_device_bias.md').write_text(f'''# Data integrity and acquisition bias

Status: COMPLETED for metadata, retained byte hashes, and frozen group overlaps. BLOCKED for patient/hospital/visit independence and historical near-duplicate verification.

Raw: {len(meta):,}; retained: {len(fold):,}; splits: {result['split_counts']}. All retained SHA-256 match the pre-existing fold manifest. Group/path/byte overlaps: `{overlap}`. Source-file overlaps do not establish patient or visit identities. Missing dedup audit prevents independently certifying the previously claimed 28,736 redundant and 66,085 label-conflicting exclusions. These counts are historical documentation only.

Metadata coverage across all {len(meta):,} sidecars: `{coverage}`. No dog/patient/visit/hospital/session ID is released. Date, breed, age and filename prefix are not substituted for patient identities. Visual grouping cannot prove dog-level independence. The pHash candidate graph is absent; the present byte/group checks do not certify that all unknown near duplicates were found.

Class × device counts: `results/tables/raw_class_device.csv` and `retained_class_device.csv`. Uncorrected Cramér's V={cramers:.6f}, chi-square={chi2:.3f}, p={p:.6g}; large sample size and related observations limit significance interpretation. Normal device counts: `{tab.loc['Normal'].to_dict()}`. Device-only class probabilities are fitted on the fresh TRAIN split and evaluated on TEST: `{bundle['metrics']}`. This baseline quantifies association, not causal removal of device bias. Neural subgroup evaluation will use only classes with true examples on the device, with explicit class counts; common supported classes will be reported separately.

The changed cohort/protocol requires fresh training. Historical performance is not transferred to it. Byte duplicates have been checked; decode/pixel/pHash verification is separately tracked.
''')
    status('audit','COMPLETED',artifacts=['results/audit.json','reports/00_project_audit.md'])
    status('integrity','COMPLETED',limitations=['patient identifiers unavailable','historical near-pair audit absent'])
    print('Audit complete:',result['split_counts'],'Cramers V',cramers,flush=True)
