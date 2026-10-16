"""Apply the primary authors' lesion-disjoint mapping to official64 rasters."""
import hashlib
import json

import numpy as np
import pandas as pd
import torch

from .common import ROOT,sha256,write_json,status
from .external import prepare
from .training import train_one


def run(args):
    torch.set_num_threads(4)
    original_npz,classes=prepare('dermamnist',64)
    # NpzFile indexing re-decompresses an entire split. Materialize each array
    # once before the row-level identity/label mapping (only ~123 MB at64).
    original={key:original_npz[key] for key in original_npz.files}
    mapping_root=ROOT/'results/external_data/derma_mapping'
    lookup=pd.read_csv(mapping_root/'dermamnist_split_info.csv')
    corrected=pd.read_csv(mapping_root/'combined_metadata_corrected-HAM10000_corrected.csv')
    source=pd.read_csv(mapping_root/'combined_metadata_original.csv')
    assert len(lookup)==len(corrected)==10015 and not lookup.image_id.duplicated().any()
    assert corrected.groupby('lesion_id').split.nunique().max()==1
    # Check the published original mapping against every actual NPZ label.
    labels={'akiec':0,'bcc':1,'bkl':2,'df':3,'mel':4,'nv':5,'vasc':6}
    joined=lookup.merge(source,on='image_id',suffixes=('_official','_metadata'),validate='one_to_one')
    assert (joined.split_official==joined.split_metadata).all()
    for row in joined.itertuples(index=False):
        assert int(original[row.split_official+'_labels'][row.index,0])==labels[row.dx]
    joined=corrected.merge(lookup,on='image_id',suffixes=('_corrected','_official'),validate='one_to_one')
    data={};frames={}
    for split in ['train','val','test']:
        part=joined[joined.split_corrected==split].reset_index(drop=True)
        data[split+'_images']=np.stack([original[r.split_official+'_images'][r.index] for r in part.itertuples(index=False)])
        data[split+'_labels']=part.dx.map(labels).to_numpy()[:,None]
        part['path']=['dermamnist_c64/'+split+'/'+str(i) for i in range(len(part))]
        frames[split]=part
        part.to_csv(ROOT/'results/external_data'/('dermamnist_c64_'+split+'_identity.csv'),index=False)
    overlap={a+'_'+b:len(set(frames[a].lesion_id)&set(frames[b].lesion_id)) for a,b in [('train','val'),('train','test'),('val','test')]}
    assert not any(overlap.values())
    original_lesion_overlap=int((source.groupby('lesion_id').split.nunique()>1).sum())
    prov=dict(source='https://github.com/kakumarabhishek/Corrected-Skin-Image-Datasets',
        mapping='combined_metadata_corrected-HAM10000_corrected.csv',mapping_sha256=sha256(mapping_root/'combined_metadata_corrected-HAM10000_corrected.csv'),
        official_mapping_sha256=sha256(mapping_root/'dermamnist_split_info.csv'),
        raster_source='official MedMNIST+64; primary author mapping reapplied locally',
        license='CC BY-NC4.0 (original images); Apache2.0 (analysis code)',
        split_sizes={s:len(f) for s,f in frames.items()},resolution=64,
        lesion_overlap=overlap,original_standard_split_leaking_lesions=original_lesion_overlap,
        all_10015_npz_labels_match_published_image_mapping=True,
        patient_independence_established=False,lesion_group_split_verified=True,
        clinical_canine_external_validation=False)
    write_json(ROOT/'results/external_data/dermamnist_c64_provenance.json',prov)
    print('Corrected lesion-disjoint dataset verified',prov,flush=True)
    args.dataset='dermamnist_c64'
    if args.train:
        for model in args.models:
            for seed in args.seeds:
                summary=train_one(args,model,seed,(data,classes))
                directory=(ROOT/summary['checkpoint']).parent
                # Model inference stays unchanged; attach verified lesion cluster IDs.
                for split in ['val','test']:
                    path=directory/(split+'_predictions.csv');f=pd.read_csv(path,keep_default_na=False)
                    identities=frames[split].set_index('path')
                    assert set(f.path)==set(identities.index)
                    f['group_id']=f.path.map(identities.lesion_id);f.to_csv(path,index=False)
    from .analysis import aggregate_new_runs
    aggregate_new_runs(args)
    status('derma_corrected','COMPLETED' if args.train else 'COMPLETED',provenance=prov)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:3');p.add_argument('--train',action='store_true')
    args=p.parse_args();args.config='configs/external64.json';args.phase='external';args.dataset='dermamnist_c64'
    args.models=['psan','gfnet','afno'];args.seeds=[42,43,44];args.seed=42;args.epochs=None;args.resume=True
    args.atom='gabor';args.atom_dropout=.2;args.atoms=16;args.init='morlet';args.axis_aligned=False;args.fixed_phase=False
    run(args)
