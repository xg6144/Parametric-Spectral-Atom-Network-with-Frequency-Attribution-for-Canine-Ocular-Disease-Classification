"""Create an isolated local replay workspace without replacing executed outputs."""
import argparse
import json
import shutil
from pathlib import Path

from .common import ROOT,sha256,write_json


def create(destination):
    dest=Path(destination).resolve()
    if dest.exists():raise FileExistsError('Reproduction destination must be new: '+str(dest))
    snapshot=ROOT/'revision_history/20261013_round2_training_source'
    frozen=json.loads((snapshot/'manifest.json').read_text())
    for name,record in frozen.items():
        if sha256(snapshot/name)!=record['sha256']:raise ValueError('Frozen source hash changed')
    dest.mkdir(parents=True)
    for p in ROOT.glob('*.py'):shutil.copy2(p,dest/p.name)
    for folder in ['experiments','models','train','eval','utils','data']:
        for p in (ROOT/folder).rglob('*.py'):
            relative=p.relative_to(ROOT);target=dest/relative;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(p,target)
    for name in frozen:
        target=dest/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(snapshot/name,target)
    # Only immutable local input references are shared. New checkpoints/logs stay here.
    for name in ['data_raw','cv_splits']:(dest/name).symlink_to(ROOT/name,target_is_directory=True)
    for folder in ['results/tables','results/status','results/experiment_logs','reports']:
        (dest/folder).mkdir(parents=True,exist_ok=True)
    for name in ['revision_split.csv','corrected_integrity.json']:
        (dest/'results'/name).symlink_to(ROOT/'results'/name)
    (dest/'results/cache').symlink_to(ROOT/'results/cache',target_is_directory=True)
    write_json(dest/'REPRODUCTION_PROVENANCE.json',dict(
        original_workspace=str(ROOT),frozen_snapshot=str(snapshot),
        protocol_sha256=sha256(dest/'configs/validation_tuning.json'),
        frozen_sources={name:record['sha256'] for name,record in frozen.items()},
        input_access='local read references; no data export or copying',
        output_isolation='new results/tuning and logs; original results untouched'))
    print('Created isolated frozen-source workspace',dest,flush=True)
    return dest


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--destination',required=True)
    args=parser.parse_args();create(args.destination)
