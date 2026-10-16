"""Replay diagnostic baseline logits and quantify batch/backend numerical differences."""
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .analysis import read_new
from .common import ROOT,CLASSES,ManifestDataset,infer,load_model,prediction_csv,seed_all,sha256,write_json


def run():
    torch.set_num_threads(4);seed_all(42);device='cuda:0'
    checkpoint=ROOT/'results/tuning/final/canine_psan_c2_s42_final/best.pth'
    directory=ROOT/'results/attribution'/sha256(checkpoint)[:12]
    frame=pd.read_csv(directory/'sample_manifest.csv',keep_default_na=False)
    model,_,_=load_model(checkpoint,device)
    records=json.loads((directory/'atom_values.json').read_text())
    expected={r['path']:r['target'] for r in records}
    original,prob,logits,_=read_new(checkpoint.with_name('test_predictions.csv'))
    positions=original.set_index('path').index.get_indexer(frame.path)
    results=[]
    original_tf32=torch.backends.cudnn.allow_tf32
    try:
        for enabled,batch in [(original_tf32,8),(original_tf32,128),(False,8),(False,128)]:
            torch.backends.cudnn.allow_tf32=enabled
            loader=DataLoader(ManifestDataset(frame),batch_size=batch,num_workers=2,shuffle=False)
            output,y=infer(model,loader,device)
            target=np.asarray([expected[p] for p in frame.path]);pred=output.argmax(1)
            name=f'baseline_b{batch}_cudnn_tf32_{enabled}'
            prediction_csv(directory/(name+'.csv'),frame,output,y,CLASSES)
            results.append(dict(batch=batch,cudnn_allow_tf32=enabled,
                matches_actual_attribution_targets=int((pred==target).sum()),images=len(frame),
                differs_from_saved_full_test=int((pred!=prob[positions].argmax(1)).sum()),
                max_logit_difference_from_saved_full_test=float(np.abs(output-logits[positions]).max()),
                source=str((directory/(name+'.csv')).relative_to(ROOT))))
    finally:torch.backends.cudnn.allow_tf32=original_tf32
    write_json(directory/'baseline_replay.json',dict(results=results,checkpoint_sha256=sha256(checkpoint),
        original_cudnn_allow_tf32=original_tf32,matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        interpretation='controlled baseline numerical diagnostic only; no checkpoint or hyperparameter selection',
        source_code_sha256=sha256(__file__)))
    print(results,flush=True)


if __name__=='__main__':run()
