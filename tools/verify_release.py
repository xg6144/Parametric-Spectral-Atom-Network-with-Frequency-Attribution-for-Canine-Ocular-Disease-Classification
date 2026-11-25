"""Check the release files and frozen sources using NumPy, pandas and scikit-learn.

1. Every source file whose SHA-256 a protocol in configs/ records is unchanged: it matches the released file or, for
   the round-2 protocols, the frozen round-2 sources in revision_history/20261013_round2_training_source/.
2. Recompute Macro F1 for the prediction files present in the release.
3. If a local audited split is present, check that each image, hash and group belongs to a single fold.

    python tools/verify_release.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / 'revision_history/20261013_round2_training_source'


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def check_manifest():
    manifest = ROOT / 'MANIFEST.csv'
    files = set()
    failures = []
    with manifest.open(newline='') as stream:
        for row in csv.DictReader(stream):
            relative = row['path']
            path = ROOT / relative
            if relative in files or not path.resolve().is_relative_to(ROOT) or path == manifest:
                failures.append(f'invalid manifest path: {relative}')
                continue
            files.add(relative)
            if not path.is_file():
                failures.append(f'missing release file: {relative}')
            elif path.stat().st_size != int(row['bytes']) or sha256(path) != row['sha256']:
                failures.append(f'release file differs from its manifest: {relative}')
    print(f'Manifest: {len(files)} files checked.')
    return failures


def fingerprints():
    """Source digests recorded in the protocols: {(protocol, file): sha256}."""
    found = {}

    def walk(value, config):
        if isinstance(value, dict):
            for key, item in value.items():
                if isinstance(item, str) and re.fullmatch(r'[0-9a-f]{64}', item) and re.search(r'\.(py|json)$', key):
                    found[(config, key)] = item
                walk(item, config)
        elif isinstance(value, list):
            for item in value:
                walk(item, config)

    for path in sorted((ROOT / 'configs').glob('*.json')):
        walk(json.loads(path.read_text()), path.relative_to(ROOT).as_posix())
    return found


def check_sources():
    failures, frozen = [], 0
    manifest = json.loads((FROZEN / 'manifest.json').read_text())
    for name, record in manifest.items():
        if sha256(FROZEN / name) != record['sha256']:
            failures.append(f'frozen round-2 source changed: {name}')
    for (config, name), digest in sorted(fingerprints().items()):
        if (ROOT / name).is_file() and sha256(ROOT / name) == digest:
            continue
        if (FROZEN / name).is_file() and sha256(FROZEN / name) == digest:
            frozen += 1
            continue
        failures.append(f'{config}: {name} does not match the recorded SHA-256')
    total = len(fingerprints())
    print(f'Sources: {total - frozen - len(failures)} of {total} recorded digests match the released files and '
          f'{frozen} the frozen round-2 sources.')
    return failures


def macro_f1(path, label, prediction, prefix):
    frame = pd.read_csv(path, usecols=lambda c: c in (label, prediction) or c.startswith(prefix))
    classes = sum(c.startswith(prefix) for c in frame.columns)
    return f1_score(frame[label], frame[prediction], labels=np.arange(classes), average='macro', zero_division=0)


def check_predictions():
    failures, checked = [], 0
    pairs = [(p, p.with_name('test_metrics.json'), 'label', 'prob::')
             for p in sorted((ROOT / 'results').rglob('test_predictions.csv.gz'))]
    pairs += [(p, p.with_name(p.name.replace('_predictions.csv.gz', '_metrics.json')), 'label', 'prob::')
              for p in sorted((ROOT / 'results/cross_validation').glob('*_out_of_fold_predictions.csv.gz'))]
    pairs += [(p, p.with_name(p.name.replace('_predictions.csv.gz', '_eval_metrics.json')), 'true_label', 'prob_')
              for p in sorted((ROOT / 'eval_results').glob('*_predictions.csv.gz'))]
    for predictions, metrics, label, prefix in pairs:
        if not metrics.is_file():
            failures.append(f'missing metrics for {predictions.relative_to(ROOT)}')
            continue
        stored = json.loads(metrics.read_text())['metrics']['macro_f1']
        recomputed = macro_f1(predictions, label, 'pred_label', prefix)
        checked += 1
        if abs(recomputed - stored) > 1e-12:
            failures.append(f'{predictions.relative_to(ROOT)}: Macro F1 {recomputed!r} != stored {stored!r}')
    print(f'Predictions: Macro F1 recomputed for {checked} prediction files.')
    return failures


def check_split():
    path = ROOT / 'results/corrected_sgkf5.csv'
    if not path.is_file():
        path = ROOT / 'results/revision_split.csv'
        if not path.is_file():
            print('Split: check available after local audit (see Data in README.md).')
            return []
    split = pd.read_csv(path, keep_default_na=False)
    failures = []
    for column in ['path', 'sha256']:
        if split[column].duplicated().any():
            failures.append(f'split manifest: duplicated {column}')
    for column in ['group_id', 'visual_group']:
        if (split.groupby(column).fold.nunique() > 1).any():
            failures.append(f'split manifest: a {column} spans several folds')
    sizes = split.fold.map(lambda fold: {1: 'test', 2: 'validation'}.get(fold, 'training')).value_counts()
    print('Split: ' + ', '.join(f'{name} {sizes[name]:,}' for name in ['training', 'validation', 'test'])
          + f' images in {split.group_id.nunique():,} visual groups.')
    return failures


def main():
    failures = check_manifest() + check_sources() + check_predictions() + check_split()
    for failure in failures:
        print('FAILED:', failure)
    print('All checks passed.' if not failures else f'{len(failures)} checks failed.')
    sys.exit(1 if failures else 0)


if __name__ == '__main__':
    main()
