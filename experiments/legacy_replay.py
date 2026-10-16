"""Recovered archival test filenames are forensic evidence, not new holdout."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from evaluation_metrics import compute_metric_bundle
from .common import ROOT,CLASSES,ManifestDataset,build_model,infer,load_model,sha256,write_json


def run():
    torch.set_num_threads(4);rows=[]
    targets=[('psan','psan_ti_M16_gabor_morlet_ad02'),('gfnet','gfnet_tiny')]
    for name,tag in targets:
        path=ROOT/'eval_results'/(tag+'_predictions.csv');frame=pd.read_csv(path)
        recovered=[]
        for row in frame.itertuples(index=False):
            basename=Path(row.path).name.split('__',1)[-1]
            raw=str(Path(row.true_class)/basename)
            if not (ROOT/'data_raw'/raw).exists():raise FileNotFoundError('Unrecoverable archival input '+raw)
            recovered.append(raw)
        manifest=pd.DataFrame({'path':recovered,'label':frame.true_label,'archival_path':frame.path})
        if manifest.path.duplicated().any():raise ValueError('Recovered paths not unique')
        manifest.to_csv(ROOT/'results'/('legacy_'+name+'_recovered_test.csv'),index=False)
        metrics=json.loads((ROOT/'eval_results'/(tag+'_eval_metrics.json')).read_text())
        checkpoint=ROOT/metrics['config']['ckpt_path']
        if name=='psan':
            model,_,source=load_model(checkpoint,'cuda:3')
            for block in model.blocks:
                f=block.filter
                f.V=torch.linspace(0,torch.pi,f.cfg.w//2+1,device='cuda:3')[None].expand(f.cfg.h,-1).contiguous()
        else:
            ckpt=torch.load(checkpoint,map_location='cpu',weights_only=False)
            source=metrics['ckpt_info']['weight_source']
            model=build_model(name);model.load_state_dict(ckpt[source],strict=True);model=model.to('cuda:3').eval()
        logits,y=infer(model,DataLoader(ManifestDataset(manifest),batch_size=128,num_workers=4),'cuda:3')
        p=torch.as_tensor(logits).softmax(-1).numpy();stored=frame.filter(regex='^prob_').to_numpy(dtype=float)
        agreement=float((p.argmax(1)==stored.argmax(1)).mean());difference=float(np.max(np.abs(p-stored)))
        bundle=compute_metric_bundle(y,p,CLASSES)
        np.savez_compressed(ROOT/'results'/('legacy_'+name+'_replayed_probabilities.npz'),probabilities=p,labels=y)
        row=dict(model=name,images=len(y),checkpoint_sha256=sha256(checkpoint),weight_source=source,
            max_probability_abs_difference=difference,stored_argmax_agreement=agreement,
            replay_macro_f1=bundle['metrics']['macro_f1'],stored_macro_f1=metrics['metrics']['macro_f1'],
            replay_macro_auc=bundle['metrics']['macro_auc_ovr'],stored_macro_auc=metrics['metrics']['macro_auc_ovr'],
            scope='raw filenames recovered; numerical consistency check, not certification of historical split/patient independence')
        rows.append(row);print(row,flush=True)
    write_json(ROOT/'results/legacy_checkpoint_replay.json',rows)


if __name__=='__main__':run()
