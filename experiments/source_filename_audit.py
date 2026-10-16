"""Validate released original-image filenames with correct sidecar alignment."""
from pathlib import Path

import pandas as pd

from .common import ROOT,sha256,write_json


def run():
    meta=pd.read_csv(ROOT/'results/raw_metadata.csv',keep_default_na=False)
    active=pd.read_csv(ROOT/'results/revision_split.csv',keep_default_na=False)
    frame=active[['path','split','group_id']].copy()
    frame['sidecar_key']=frame.path.map(lambda s:str(Path(s).with_suffix('')))
    frame=frame.merge(meta[['path','file_name']].rename(columns={'path':'sidecar_key'}),
                      on='sidecar_key',how='left',validate='one_to_one')
    if frame.file_name.isna().any():raise ValueError('Original-filename metadata alignment incomplete')
    frame['normalized_filename']=frame.file_name.astype(str).str.replace('\\','/',regex=False).map(lambda s:Path(s).name.strip())
    groups=frame[frame.normalized_filename!=''].groupby('normalized_filename')
    duplicate=groups.size();duplicate=duplicate[duplicate>1]
    crossing=groups.split.nunique();crossing=crossing[crossing>1]
    record=dict(images=len(frame),metadata_coverage=int(frame.normalized_filename.ne('').sum()),
        distinct_nonempty_filenames=groups.ngroups,duplicate_filename_keys=len(duplicate),
        duplicate_filename_rows=int(duplicate.sum()),cross_split_filename_keys=len(crossing),
        alignment='raw metadata paths identify JSON sidecar stems; active image paths include extensions, normalized before a validated1to1 join',
        limitation='original-image filenames are released proxies; uniqueness does not establish dog/visit/hospital identity',
        source_metadata_sha256=sha256(ROOT/'results/raw_metadata.csv'),active_split_sha256=sha256(ROOT/'results/revision_split.csv'),
        source_code_sha256=sha256(__file__))
    write_json(ROOT/'results/source_filename_integrity.json',record)
    if len(crossing):
        frame[frame.normalized_filename.isin(crossing.index)].to_csv(ROOT/'results/source_filename_cross_split.csv',index=False)
        raise ValueError('Repeated released original filename crosses frozen splits')
    print('Original-filename coverage and overlap verified',record,flush=True)


if __name__=='__main__':run()
