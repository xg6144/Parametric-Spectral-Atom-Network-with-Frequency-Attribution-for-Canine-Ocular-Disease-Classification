"""Rebuild results/revision_split.csv, the train/validation/test manifest that the training and analysis code reads.

The manifest joins the released split (results/corrected_sgkf5.csv) with per-image metadata from the JSON records of
the AI-Hub dataset (capture date, device, file name, breed, sex, age and eye side), which is why it is not distributed
with this repository. The script rebuilds it from your copy of the dataset in data_raw/ and accepts the result only
if it matches the SHA-256 recorded in results/corrected_integrity.json, which the code checks before it uses the split.

    python tools/restore_split_manifest.py [--check-images]
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

COLUMNS = ['path', 'class', 'label', 'group_id', 'fold', 'sha256', 'visual_group', 'split', 'device', 'date_time',
           'file_name', 'breed', 'gender', 'age', 'eye_position']


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description='Rebuild results/revision_split.csv from the AI-Hub JSON records.')
    parser.add_argument('--check-images', action='store_true', help='also check every retained image against its SHA-256')
    args = parser.parse_args()
    records = sorted((ROOT / 'data_raw').glob('*/*.json'))
    if not records:
        raise SystemExit('No JSON records in data_raw/<class>/; see "Data" in README.md.')
    from experiments.audit import metadata_record  # the reader that produced the released manifest

    with ThreadPoolExecutor(max_workers=16) as pool:
        metadata = pd.DataFrame(pool.map(metadata_record, records))
    # The same steps as experiments/integrity.py.
    meta = pd.read_csv(io.StringIO(metadata.to_csv(index=False)), keep_default_na=False)
    split = pd.read_csv(ROOT / 'results/corrected_sgkf5.csv', keep_default_na=False)
    split['split'] = np.where(split.fold == 1, 'test', np.where(split.fold == 2, 'val', 'train'))
    meta['meta_path'] = meta.path
    split['meta_path'] = split.path.map(lambda p: str(Path(p).with_suffix('')))
    merged = split.merge(meta.drop(columns=['path', 'class']), on='meta_path', validate='one_to_one')
    data = merged[COLUMNS].to_csv(index=False).encode()
    expected = json.loads((ROOT / 'results/corrected_integrity.json').read_text())['revision_manifest_sha256']
    digest = hashlib.sha256(data).hexdigest()
    if digest != expected:
        raise SystemExit(f'The rebuilt manifest ({len(merged):,} of {len(split):,} images matched to a JSON record) has '
                         f'SHA-256 {digest}, not the recorded {expected}; data_raw/ does not hold the same records.')
    if args.check_images:
        image = lambda p: sha256(ROOT / 'data_raw' / p) if (ROOT / 'data_raw' / p).is_file() else None
        with ThreadPoolExecutor(max_workers=8) as pool:
            actual = list(pool.map(image, merged.path))
        wrong = int((np.array(actual, dtype=object) != merged.sha256.to_numpy(dtype=object)).sum())
        if wrong:
            raise SystemExit(f'{wrong:,} of {len(merged):,} retained images are missing or differ from their recorded '
                             'SHA-256.')
        print(f'All {len(merged):,} retained images match their recorded SHA-256.')
    (ROOT / 'results/revision_split.csv').write_bytes(data)
    print(f'Wrote results/revision_split.csv ({len(merged):,} images; SHA-256 {digest}).')


if __name__ == '__main__':
    main()
