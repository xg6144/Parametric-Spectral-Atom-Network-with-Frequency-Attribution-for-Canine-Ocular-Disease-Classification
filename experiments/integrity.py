"""Freeze current independently reconstructed audit and union visual graphs."""
import json
import platform
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
import torch
from scipy.stats import chi2_contingency

from prepare_stratified_group_folds import create_fold_manifest
from evaluation_metrics import compute_metric_bundle
from .common import ROOT,CLASSES,sha256,status,write_json


def run(args):
    status('integrity','IN PROGRESS')
    audit_root=ROOT/'results/reconstructed_dedup'
    d=pd.read_csv(audit_root/'dedup_manifest.csv',keep_default_na=False)
    old=pd.read_csv(ROOT/'cv_splits/stratified_group_5fold.csv',keep_default_na=False)
    near=pd.read_csv(audit_root/'near_duplicate_pairs.csv',keep_default_na=False)
    parent=list(range(len(d)));path_index={p:i for i,p in enumerate(d.path)}
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:parent[max(a,b)]=min(a,b)
    for frame,key in [(d,'visual_group'),(old,'group_id')]:
        first={}
        for row in frame.itertuples(index=False):
            group=getattr(row,key)
            if not group:continue
            index=path_index[row.path]
            if group in first:union(index,first[group])
            else:first[group]=index
    d['reconstructed_visual_group']=d.visual_group
    d['visual_group']=[f'R{find(i):06d}' for i in range(len(d))]
    merged=ROOT/'results/corrected_dedup_manifest.csv'
    if merged.exists():
        existing=pd.read_csv(merged,keep_default_na=False)
        if not existing.equals(d):raise ValueError('Existing corrected manifest differs; immutable provenance required')
    else:d.to_csv(merged,index=False)
    records,fold_summary=create_fold_manifest(
        audit_manifest=merged,data_root=ROOT/'data_raw',
        output_manifest=ROOT/'results/corrected_sgkf5.csv',
        output_summary=ROOT/'results/corrected_sgkf5_summary.json',
        n_splits=5,seed=42)
    f=pd.read_csv(ROOT/'results/corrected_sgkf5.csv',keep_default_na=False)
    f['split']=np.where(f.fold==1,'test',np.where(f.fold==2,'val','train'))
    meta=pd.read_csv(ROOT/'results/raw_metadata.csv',keep_default_na=False)
    meta['meta_path']=meta.path
    from pathlib import Path
    f['meta_path']=f.path.map(lambda p:str(Path(p).with_suffix('')))
    f=f.merge(meta.drop(columns=['path','class']),on='meta_path',validate='one_to_one')
    cols=['path','class','label','group_id','fold','sha256','visual_group','split','device','date_time','file_name','breed','gender','age','eye_position']
    f=f[cols]
    active=ROOT/'results/revision_split.csv'
    if active.exists() and (ROOT/'results/corrected_integrity.json').exists():
        if not pd.read_csv(active,keep_default_na=False).equals(f):raise ValueError('Active split changed')
    f.to_csv(active,index=False)
    retained=d[d.path.isin(f.path)]
    assert len(retained)==len(f)==104037 and retained.status.eq('keep').all()
    assert not retained.sha256.duplicated().any() and not retained.pixel_sha256.duplicated().any()
    pair=near[near.left_path.isin(f.path)&near.right_path.isin(f.path)].copy()
    mapping=f.set_index('path');pair['left_split']=pair.left_path.map(mapping.split);pair['right_split']=pair.right_path.map(mapping.split)
    cross=int((pair.left_split!=pair.right_split).sum());verified=int(pair.verified.astype(int).sum())
    assert cross==verified==0
    overlaps={}
    for a,b in combinations(['train','val','test'],2):
        aa,bb=f[f.split==a],f[f.split==b]
        overlaps[a+'_'+b]={'paths':len(set(aa.path)&set(bb.path)),
            'visual_groups':len(set(aa.group_id)&set(bb.group_id)),
            'byte_sha256':len(set(aa.sha256)&set(bb.sha256))}
    coverage={k:int((meta[k]!='').sum()) for k in ['file_name','date_time','device','dog_id','patient_id','visit_id','hospital_id','session_id']}
    tab=pd.crosstab(f['class'],f.device);tab.to_csv(ROOT/'results/tables/corrected_class_device.csv')
    chi2,p,_,_=chi2_contingency(tab);v=float(np.sqrt(chi2/len(f)/min(tab.shape[0]-1,tab.shape[1]-1)))
    train,test=f[f.split=='train'],f[f.split=='test']
    dp=pd.crosstab(train.device,train.label).reindex(columns=range(11),fill_value=0);dp=dp.div(dp.sum(1),axis=0)
    probability=np.stack([dp.loc[x].to_numpy() for x in test.device])
    bundle=compute_metric_bundle(test.label.to_numpy(),probability,CLASSES);write_json(ROOT/'results/device_only_baseline_corrected.json',bundle)
    reconstruction_path=ROOT/'results/integrity_reconstruction.json'
    if not reconstruction_path.exists():
        historical=d[d.path.isin(old.path)]
        historical_pairs=near[near.left_path.isin(old.path)&near.right_path.isin(old.path)]
        old_index=old.set_index('path')
        old_split=np.where(old.fold==1,'test',np.where(old.fold==2,'val','train'))
        split_lookup=dict(zip(old.path,old_split))
        reconstruct=dict(current_retained=int(d.status.eq('keep').sum()),frozen_retained=len(old),
            frozen_pixel_duplicate_rows=int(historical.pixel_sha256.duplicated().sum()),
            frozen_retained_current_status=historical.status.value_counts().to_dict(),
            retained_candidate_pairs=len(historical_pairs),
            retained_verified_pairs=int(historical_pairs.verified.astype(int).sum()),
            cross_split_candidate_pairs=int((historical_pairs.left_path.map(split_lookup)!=historical_pairs.right_path.map(split_lookup)).sum()),
            cross_group_candidate_pairs=int((historical_pairs.left_path.map(old_index.group_id)!=historical_pairs.right_path.map(old_index.group_id)).sum()),
            frozen_only_paths=len(set(old.path)-set(d[d.status=='keep'].path)),
            reconstruction_only_paths=len(set(d[d.status=='keep'].path)-set(old.path)))
        write_json(reconstruction_path,reconstruct)
    reconstruct=json.loads(reconstruction_path.read_text())
    summary=json.loads((audit_root/'summary.json').read_text())
    result=dict(raw_images=len(d),retained_images=len(f),retained_groups=int(f.group_id.nunique()),
        split_counts=f.split.value_counts().to_dict(),overlap=overlaps,
        retained_byte_duplicate_rows=int(retained.sha256.duplicated().sum()),
        retained_pixel_duplicate_rows=int(retained.pixel_sha256.duplicated().sum()),
        retained_candidate_pairs=len(pair),retained_verified_pairs=verified,cross_split_candidate_pairs=cross,
        metadata_coverage=coverage,normal_device_counts=tab.loc['Normal'].to_dict(),cramers_v=v,
        device_only_metrics=bundle['metrics'],dedup_policy_disagreement=reconstruct,
        revised_manifest_source=str(merged.relative_to(ROOT)),dedup_manifest_sha256=sha256(merged),
        reconstructed_audit_sha256=sha256(audit_root/'dedup_manifest.csv'),
        revision_manifest_sha256=sha256(active),patient_independence_established=False,
        group_method='union of current raw visual groups and previous retained visual groups; StratifiedGroupKFold seed42')
    write_json(ROOT/'results/corrected_integrity.json',result)
    inventory=f.groupby('class').size().to_dict()
    with (ROOT/'reports/00_project_audit.md').open('a') as file:
        file.write(f'''\n## Independent full audit supersedes the initial frozen-cohort check\n\nAll {len(d):,} raw images were decoded, byte/pixel hashed and pHash screened; decode errors=0. Current executed policy retains {len(f):,}, excluding {summary['result']['status_counts']}. Compared with the older104053 manifest: {reconstruct}. Three VERIFIED cross-split near pairs made the first fresh-training protocol invalid. All three interrupted runs were stopped before final test inference and preserved in `results/superseded_training/20261012_frozen_v1/`; their metrics must not enter the final table. Differences in candidate/ORB matching under present libraries are recorded, not silently reconciled by forcing historical counts. Fresh models restart from scratch under the current independently verified policy and immutable graph union. Initial report values refer to the superseded audit; active evidence is `corrected_integrity.json`.\n''')
    (ROOT/'reports/01_data_integrity_and_device_bias.md').write_text(f'''# Independently reconstructed data integrity and acquisition bias

Status: COMPLETED for actual raw/retained hash/pixel/pHash audit, visual graph split and device analysis. BLOCKED for dog/visit/hospital independence (IDs not released).

Current source audit: {len(d):,} images; no decoding errors; exact byte groups={summary['exact']['exact_file_groups']}; exact pixel groups={summary['exact']['exact_pixels_groups']}; pHash threshold5 candidates={summary['near']['candidate_pairs']}, independently verified={summary['near']['verified_pairs']}. Current exclusions={summary['result']['status_counts']}; current retained={len(f):,}. Class counts={inventory}. Full per-image/near-pair/cache artifacts and parameters are in `results/reconstructed_dedup/`. Original files are untouched. Byte hashes support the same identity purpose as MD5; MD5 was additionally computed for all older retained images. pHash alone never authorizes removal.

The original104053 manifest fails the new near-pair audit: {reconstruct}. It contains three verified near-pair edges crossing the first revision train/val/test split. Initial training was stopped at partial epochs and quarantined before test inference; no such checkpoint is valid final performance evidence. Current independently executed exclusions differ from historical README counts. The unavailable historical audit cannot be reconstructed bit-for-bit under changed imaging/feature libraries, so the current104037 cohort is frozen as a new experiment. Original and reconstructed grouping edges are unioned to retain conservative historical connections.

Current split sizes={result['split_counts']}; groups={result['retained_groups']}. Cross-split path/group/byte overlap={overlaps}; retained pixel duplicates=0; retained VERIFIED near pairs=0; all retained candidate-pair edges crossing splits=0 ({len(pair)} retained candidate edges tested). These are algorithm-dependent checks, not proof that every possible visual duplicate or repeated dog was detected.

Metadata coverage={coverage}. No dog/patient/visit/hospital/session identity exists. Dates, demographics, original filenames and D-prefixes cannot prove independent dogs or visits. These fields are descriptive only. Patient-level independence is not claimed.

Normal capture counts={result['normal_device_counts']}; Cramér's V={v:.6f} (unadjusted descriptive association). Chi-square p underflows numerical precision; no causal claim follows. Device-only posterior is fitted strictly on TRAIN and evaluated on TEST: {bundle['metrics']}. `corrected_class_device.csv` contains class×device coverage. Neural subgroup evaluation uses supported true classes plus a separate common-supported-class analysis; absent device classes are never assigned artificial recall/F1 values. Association is measured, not removed. The evaluation still cannot establish veterinary external clinical validity.

Reproducible command: `python run_experiments.py --phase audit`. Active split SHA256={result['revision_manifest_sha256']}; current raw audit SHA256={result['reconstructed_audit_sha256']}.
''')
    config=ROOT/'configs/revision.json';cfg=json.loads(config.read_text());cfg['revision_protocol']='independently_reconstructed_union_visual_groups_train345_val2_test1_v2';config.write_text(json.dumps(cfg,indent=2)+'\n')
    status('audit','COMPLETED',full_raw_images=len(d),retained_images=len(f))
    status('integrity','COMPLETED',patient_independence=False,cross_split_candidate_pairs=0,
           source='results/corrected_integrity.json')
    print('Corrected frozen split ready',result,flush=True)
