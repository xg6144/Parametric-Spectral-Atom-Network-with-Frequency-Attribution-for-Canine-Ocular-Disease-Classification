"""Additional measurements derived only from saved actual interventions."""
import json

import numpy as np
import pandas as pd

from .analysis import read_new
from .common import ROOT,sha256,write_json
from .faithfulness import mean_ci


def run(directory=None):
    directory=directory or ROOT/'results/attribution/3361a540d47d'
    provenance=json.loads((directory/'provenance.json').read_text())
    checkpoint=ROOT/provenance['checkpoint'];run_dir=checkpoint.parent
    summary=json.loads((run_dir/'summary.json').read_text())
    assert summary['checkpoint_sha256']==provenance['checkpoint_sha256']==sha256(checkpoint)
    diagnostic_baseline=directory/'baseline_b8_cudnn_tf32_True.csv'
    baseline_path=diagnostic_baseline if diagnostic_baseline.exists() else run_dir/'test_predictions.csv'
    frame,prob,_,_=read_new(baseline_path)
    curve=pd.read_csv(directory/'deletion_curves.csv',keep_default_na=False)
    positions=frame.set_index('path').index.get_indexer(curve.path)
    assert (positions>=0).all()
    target=curve.target.to_numpy(dtype=int)
    assert np.array_equal(prob[positions].argmax(1),target)
    curve['baseline_target_confidence']=prob[positions,target]
    curve['confidence_drop']=curve.baseline_target_confidence-curve.confidence
    curve.to_csv(directory/'confidence_changes.csv',index=False)
    rows=[]
    for (block,method),part in curve.groupby(['block','method']):
        per_image=part.groupby(['path','group_id']).confidence_drop.mean().reset_index()
        rows.append(dict(block=block,method=method,**mean_ci(per_image.confidence_drop,per_image.group_id,42,1000)))
    pd.DataFrame(rows).to_csv(directory/'confidence_summary.csv',index=False)
    values=json.loads((directory/'atom_values.json').read_text());single=[]
    for row in values:
        a=np.asarray(row['attribution']);d=np.asarray(row['actual_logit_drops'])
        choices={'lowest_signed':int(a.argmin()),'lowest_absolute':int(np.abs(a).argmin())}
        if (a>0).any():choices['highest_positive']=int(a.argmax())
        if (a<0).any():choices['negative_lowest']=int(a.argmin())
        for method,index in choices.items():
            single.append(dict(path=row['path'],group_id=row['group_id'],block=row['block'],
                method=method,atom=index,attribution=a[index],logit_drop=d[index]))
    pd.DataFrame(single).to_csv(directory/'conditional_single_atom.csv',index=False)
    rows=[]
    for (block,method),part in pd.DataFrame(single).groupby(['block','method']):
        rows.append(dict(block=block,method=method,**mean_ci(part.logit_drop,part.group_id,42,1000)))
    pd.DataFrame(rows).to_csv(directory/'conditional_single_atom_summary.csv',index=False)
    write_json(directory/'additional_diagnostics_provenance.json',dict(
        source='saved actual finite atom/suffix interventions; no new model or invented perturbation',
        checkpoint_sha256=provenance['checkpoint_sha256'],
        raw_baseline_prediction_sha256=sha256(baseline_path),baseline_path=str(baseline_path.relative_to(ROOT)),
        negative_condition='report only images with at least one negative atom; missing negatives are excluded and n reported',
        positive_condition='report only images with a positive atom; budget one',
        confidence='original target raw probability minus perturbed raw target probability; mean over k1..16 then images',
        precision_note='use replayed diagnostic-batch baseline when available; cuDNN TF32 can change a near-boundary prediction across batch sizes, as recorded in baseline_replay.json',
        bootstrap=1000,seed=42))
    print('Derived confidence changes and conditional positive/negative removals from actual saved outputs.',flush=True)


if __name__=='__main__':run()
