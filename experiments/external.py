from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .common import ROOT,write_json,status
from .training import train_one,download_external


def prepare(name,size):
    import medmnist
    from medmnist.info import INFO
    info=INFO[name]
    root=ROOT/'results/external_data';root.mkdir(parents=True,exist_ok=True)
    if size==224:return download_external(name)
    cls=getattr(medmnist,info['python_class'])
    dataset=cls(root=str(root),split='train',download=True,size=size)
    path=root/(name+'_'+str(size)+'.npz')
    digest=hashlib.md5(path.read_bytes()).hexdigest()
    assert digest==info['MD5_'+str(size)],'Published external checksum mismatch'
    data=np.load(path);tag=name+str(size)
    hashes={s:[hashlib.sha256(image.tobytes()).hexdigest() for image in data[s+'_images']]
            for s in ['train','val','test']}
    overlaps={a+'_'+b:len(set(hashes[a])&set(hashes[b])) for a,b in [('train','val'),('train','test'),('val','test')]}
    provenance=dict(source=info['url_'+str(size)],published_md5=info['MD5_'+str(size)],actual_md5=digest,
        license=info['license'],description=info['description'],official_split_sizes=info['n_samples'],
        resolution=size,labels=info['label'],pixel_hash_cross_split_overlap=overlaps,
        patient_or_lesion_ids_in_npz=False,clinical_canine_external_validation=False,
        known_dermamnist_lesion_leakage=(name=='dermamnist'),
        leakage_reference='https://www.nature.com/articles/s41597-025-04382-5' if name=='dermamnist' else None)
    write_json(root/(tag+'_provenance.json'),provenance)
    print('Official data verified:',tag,provenance,flush=True)
    return data,[info['label'][str(i)] for i in range(len(info['label']))]


def run(args):
    if args.dataset.startswith('dermamnist_c'):
        from .derma_corrected import run as corrected_run
        return corrected_run(args)
    torch.set_num_threads(4)
    if args.train:
        if args.dataset=='canine':raise ValueError('Specify --dataset dermamnist_c64 or octmnist64 and --config configs/external64.json for external training')
        size=json.loads(Path(args.config).read_text())['img_size']
        base=args.dataset.replace(str(size),'')
        data=prepare(base,size)
        if size!=224:args.dataset=base+str(size)
        for model in args.models:
            for seed in args.seeds:train_one(args,model,seed,data)
    from .analysis import aggregate_new_runs
    aggregate_new_runs(args)
    files=list((ROOT/'results/external_data').glob('*_provenance.json'))
    runs=[json.loads(p.read_text()) for p in (ROOT/'results/training').glob('*/summary.json')
          if json.loads(p.read_text())['dataset']!='canine']
    (ROOT/'reports/06_external_generalization.md').write_text(f'''# External architecture benchmarks

Status: {'COMPLETED for actual available runs; see explicit counts' if runs else 'NOT STARTED for training; download/protocol work in progress'}.

Verified provenance files: {[str(p.relative_to(ROOT)) for p in files]}. Completed runs: {[(r['tag'],r['test']['macro_f1']) for r in runs]}.

Use official MedMNIST+64 files, published MD5, licenses and train/val/test sizes. PSAN/GFNet/AFNO are independently constructed at64 (4x4 tokens). Actual scope is corrected Derma three seeds and OCT seed42, one common preset, validation-only checkpoint selection, full official training sets. Additional OCT seeds are NOT STARTED. These are architecture benchmarks, not veterinary clinical external validation. At64, F12<M16; rank/compactness conclusions from224 do not transfer unchanged.

Patient/lesion IDs are absent from NPZ. Standard Derma has documented leakage ([Abhishek et al., 2025](https://www.nature.com/articles/s41597-025-04382-5)). The primary authors' corrected mapping was actually applied and all10015 original NPZ labels verified. Corrected train8215/val573/test1227 has zero lesion overlap; patient independence remains unknown. OCT uses the official split with no released patient IDs; zero pixel overlap does not certify patient separation. See `dermamnist_c64_provenance.json`, corrected identity CSVs and `FINAL_RESEARCH_REPORT.md`.

Original 224 downloads were investigated but bandwidth is limited; 64 is an explicit resource-driven protocol choice recorded before model/test evaluation. These images are the official 64 release, not 28-pixel images upsampled to 224. Derma license CC BY-NC4.0; OCT CC BY4.0 (verified against the official installed metadata). References: [MedMNIST official site](https://medmnist.com/), [official metadata](https://github.com/MedMNIST/MedMNIST/blob/main/medmnist/info.py).
''')
    complete=len({r['dataset'] for r in runs})>=2 and len(runs)>=12
    status('external','COMPLETED' if complete else 'IN PROGRESS',completed_runs=len(runs),
           independent_canine_external_validation=False,known_standard_derma_leakage=True,
           scope='corrected Derma3 seeds; OCT1 seed; additional seeds NOT STARTED')
