"""Descriptive agreement of two released initial severity fields, not CAM review."""
from pathlib import Path

import numpy as np
import pandas as pd

from .common import ROOT,sha256,write_json


def agreement(frame):
    part=frame[(frame.reviewer1!='') & (frame.reviewer2!='')]
    first=part.reviewer1.astype(str);second=part.reviewer2.astype(str)
    categories=sorted(set(first)|set(second));n=len(part)
    observed=float(first.eq(second).mean())
    p=first.value_counts().reindex(categories,fill_value=0).to_numpy()/n
    q=second.value_counts().reindex(categories,fill_value=0).to_numpy()/n
    expected=float(p@q)
    return dict(images=n,missing=int(len(frame)-n),disagreements=int(first.ne(second).sum()),
        observed_agreement=observed,expected_marginal_agreement=expected,
        unweighted_cohen_kappa=(observed-expected)/(1-expected) if expected<1 else None,
        categorical_terms='|'.join(categories))


def run():
    path=ROOT/'results/raw_label_annotations.csv';frame=pd.read_csv(path,keep_default_na=False)
    active=pd.read_csv(ROOT/'results/revision_split.csv',keep_default_na=False)
    keys=set(active.path.map(lambda s:str(Path(s).with_suffix('.json'))))
    retained=frame[frame.json_path.isin(keys)]
    if len(retained)!=104037:raise ValueError('Rater-field retained coverage mismatch')
    rows=[]
    for cohort,part in [('raw',frame),('retained',retained)]:
        rows.append(dict(cohort=cohort,source_disease='ALL_DESCRIPTIVE_MIXED_TERMS',**agreement(part)))
        for disease,group in part.groupby('disease'):
            rows.append(dict(cohort=cohort,source_disease=disease,**agreement(group)))
    table=pd.DataFrame(rows);table.to_csv(ROOT/'results/tables/released_rater_agreement.csv',index=False)
    overall=table[table.source_disease=='ALL_DESCRIPTIVE_MIXED_TERMS']
    if list(overall.disagreements)!=[41638,7592]:raise ValueError('Agreement disagrees with full label audit')
    write_json(ROOT/'results/released_rater_agreement_protocol.json',dict(
        annotations_sha256=sha256(path),split_sha256=sha256(ROOT/'results/revision_split.csv'),source_code_sha256=sha256(__file__),
        target='exact categorical strings in the two initial released severity fields; unweighted kappa=(observed-expected)/(1-expected)',
        limits='descriptive annotation agreement, not clinical accuracy, independence/blinding or explanation evaluation; disease-stratified terms preferred over mixed overall terms',
        final_field='reviewed/adjudicated final reference; not treated as a third independent blind rater',
        selection='raw and retained cohorts differ by duplicate/conflict policy; agreement change is not a causal quality estimate'))
    print('Released initial-rater agreement measured:',overall[['cohort','images','observed_agreement','unweighted_cohen_kappa']].to_dict('records'),flush=True)


if __name__=='__main__':run()
