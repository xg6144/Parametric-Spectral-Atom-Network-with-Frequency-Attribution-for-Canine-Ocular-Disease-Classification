"""Observed confusion patterns and error confidence; no clinical causal labels."""
import json

import numpy as np
import pandas as pd

from .analysis import read_new
from .common import ROOT,sha256,write_json

MINORITY=['Nuclear_sclerosis','Eyelid_tumor','Non_ulcerative_keratitis',
          'Pigmentary_keratitis','Epiphora']


def run():
    rows=[];device_rows=[]
    paths=[]
    for name in ['psan_gabor_ad0p2_M16_morlet','gfnet','afno']:
        path=ROOT/'results/training'/('canine_'+name+'_s42/test_predictions.csv')
        frame,p,_,classes=read_new(path);paths.append(path)
        pred=p.argmax(1);confidence=p.max(1)
        for c in MINORITY:
            index=classes.index(c);subset=frame.label.to_numpy()==index
            wrong=subset&(pred!=index);correct=subset&(pred==index)
            counts=np.bincount(pred[wrong],minlength=len(classes))
            ranked=sorted(range(len(classes)),key=lambda k:(-counts[k],k))[:2]
            rows.append(dict(model=name.split('_gabor')[0],seed=42,
                **{'class':c},support=int(subset.sum()),correct=int(correct.sum()),errors=int(wrong.sum()),
                correct_mean_max_probability=float(confidence[correct].mean()) if correct.any() else None,
                error_mean_max_probability=float(confidence[wrong].mean()) if wrong.any() else None,
                most_common_wrong_class=classes[ranked[0]],most_common_wrong_count=int(counts[ranked[0]]),
                most_common_wrong_fraction=float(counts[ranked[0]]/wrong.sum()) if wrong.any() else None,
                second_wrong_class=classes[ranked[1]],second_wrong_count=int(counts[ranked[1]])))
            for device in sorted(frame.device.unique()):
                group=subset&(frame.device.to_numpy()==device)
                if not group.any():continue
                errors=group&(pred!=index);counts=np.bincount(pred[errors],minlength=len(classes))
                top=int(counts.argmax())
                device_rows.append(dict(model=name.split('_gabor')[0],seed=42,device=device,
                    **{'class':c},support=int(group.sum()),correct=int((group&(pred==index)).sum()),
                    recall=float((group&(pred==index)).sum()/group.sum()),errors=int(errors.sum()),
                    most_common_wrong_class=classes[top] if errors.any() else None,
                    most_common_wrong_count=int(counts[top])))
    pd.DataFrame(rows).to_csv(ROOT/'results/tables/minority_error_patterns.csv',index=False)
    pd.DataFrame(device_rows).to_csv(ROOT/'results/tables/minority_device_errors.csv',index=False)
    write_json(ROOT/'results/minority_error_protocol.json',dict(seed=42,classes=MINORITY,
        source_sha256={str(p.relative_to(ROOT)):sha256(p) for p in paths},
        source_code_sha256=sha256(__file__),definition='errors are argmax != true label; top wrong classes use counts among errors',
        interpretation='descriptive conditional errors; unknown patient dependence and device class coverage remain',
        medical_causes='not inferred; expert review and lesion annotations unavailable'))


if __name__=='__main__':run()
