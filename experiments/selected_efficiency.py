"""Selected-checkpoint caching and separate process-isolated runtime session.

Preserves the earlier historical-checkpoint benchmark. Every GPU timing cell
contains one trained model, one input batch, and its optional filter cache.
"""
import argparse
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .common import ROOT,ManifestDataset,infer,load_manifest,load_model,seed_all,sha256,status,write_json
from .efficiency import measure

OUT=ROOT/'results/selected_efficiency'


def checkpoint(name):
    return json.loads((ROOT/'results/tuning/final_selected'/f'{name}.json').read_text())


def cell(name,condition,batch,device):
    torch.set_num_threads(4);seed_all(42)
    dev=torch.device(device)
    if dev.type=='cuda':torch.cuda.set_device(dev)
    s=checkpoint(name);path=ROOT/s['checkpoint']
    model,_,_=load_model(path,dev)
    if condition=='cached':model.cache_filters(s['checkpoint_sha256'])
    x=torch.randn(batch,3,224,224,device=dev)
    r=dict(model=name,condition=condition,batch=batch,device=device,precision='float32',
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        checkpoint=s['checkpoint'],checkpoint_sha256=s['checkpoint_sha256'],
        parameters=sum(p.numel() for p in model.parameters()),cpu_threads=4,
        source_code_sha256=sha256(__file__),
        **measure(model,x,dev,warmup=20 if dev.type=='cuda' else 10,iterations=100 if dev.type=='cuda' else 30))
    write_json(OUT/f'{name}_{condition}_b{batch}_{dev.type}.json',r)


def equivalence(device):
    torch.set_num_threads(4);seed_all(42)
    dev=torch.device(device)
    if dev.type=='cuda':torch.cuda.set_device(dev)
    s=checkpoint('psan');model,_,_=load_model(ROOT/s['checkpoint'],dev)
    frame=load_manifest();frame=frame[frame.split=='test'].reset_index(drop=True)
    loader=DataLoader(ManifestDataset(frame),batch_size=128,num_workers=2,pin_memory=True)
    dynamic,y=infer(model,loader,dev)
    model.cache_filters(s['checkpoint_sha256']);cached,yc=infer(model,loader,dev)
    if not np.array_equal(y,yc):raise ValueError('Cached label alignment failed')
    diff=float(np.abs(dynamic-cached).max())
    if diff>1e-5 or not np.array_equal(dynamic.argmax(1),cached.argmax(1)):
        raise ValueError('Selected full-test cache equivalence failed')
    saved=pd.read_csv((ROOT/s['checkpoint']).with_name('test_predictions.csv'))
    if not np.array_equal(saved.path,frame.path) or not np.array_equal(saved.label,y):
        raise ValueError('Selected cache test paths differ from main evaluation')
    original=saved.filter(regex=r'^logit::').to_numpy()
    np.savez_compressed(OUT/'full_test_cache_equivalence.npz',dynamic=dynamic,cached=cached,labels=y)
    record=dict(images=len(y),batch_size=128,checkpoint=s['checkpoint'],checkpoint_sha256=s['checkpoint_sha256'],
        split_sha256=sha256(ROOT/'results/revision_split.csv'),max_logit_abs_difference=diff,
        argmax_agreement=float(np.mean(dynamic.argmax(1)==cached.argmax(1))),
        accuracy_dynamic=float(np.mean(dynamic.argmax(1)==y)),accuracy_cached=float(np.mean(cached.argmax(1)==y)),
        replay_max_logit_difference=float(np.abs(dynamic-original).max()),
        replay_argmax_disagreements=int(np.sum(dynamic.argmax(1)!=original.argmax(1))),
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        test_predictions_sha256=sha256((ROOT/s['checkpoint']).with_name('test_predictions.csv')),
        arrays='results/selected_efficiency/full_test_cache_equivalence.npz',source_code_sha256=sha256(__file__))
    write_json(OUT/'cache_equivalence.json',record)
    print('Selected full-test cache equality',record,flush=True)


def main(device):
    OUT.mkdir(parents=True,exist_ok=True)
    status('selected_efficiency','IN PROGRESS')
    subprocess.run([sys.executable,'-m','experiments.selected_efficiency','--device',device,'--equivalence'],check=True)
    cells=[(name,condition,batch,device) for name in ['psan','gfnet','afno'] for batch in [1,16,64]
           for condition in (['dynamic','cached'] if name=='psan' else ['dynamic'])]
    cells += [('psan',condition,batch,'cpu') for batch in [1,16] for condition in ['dynamic','cached']]
    rows=[]
    for name,condition,batch,dev in cells:
        subprocess.run([sys.executable,'-m','experiments.selected_efficiency','--device',dev,
                        '--model',name,'--condition',condition,'--batch',str(batch)],check=True)
        path=OUT/f'{name}_{condition}_b{batch}_{torch.device(dev).type}.json'
        rows.append(json.loads(path.read_text())|dict(measurement_source=str(path.relative_to(ROOT))))
        print('Completed independent timing cell',name,condition,batch,dev,flush=True)
    pd.DataFrame(rows).to_csv(ROOT/'results/tables/selected_efficiency.csv',index=False)
    write_json(OUT/'protocol.json',dict(device=device,pytorch=torch.__version__,cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(torch.device(device)),cudnn=torch.backends.cudnn.version(),
        models='validation-selected trained seed42 final checkpoints',input='synthetic float32 224x224 RGB',
        timing='batch model-only latency; GPU20 warmups/100 synchronized calls, CPU10/30 calls',
        memory='one independent process per model/path/batch; total peak allocated MiB including model/input/cache',
        excluded='checkpoint loading, cache setup, I/O/preprocessing and upstream pretraining',
        concurrency='other local training GPUs can affect host overhead; hardware/session-specific measurement',
        original_session_preserved='results/tables/efficiency.csv',source_code_sha256=sha256(__file__)))
    status('selected_efficiency','COMPLETED',rows=len(rows),full_test_images=20808)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--device',required=True)
    parser.add_argument('--model',choices=['psan','gfnet','afno']);parser.add_argument('--condition',choices=['dynamic','cached'])
    parser.add_argument('--batch',type=int);parser.add_argument('--equivalence',action='store_true');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if args.equivalence:equivalence(args.device)
    elif args.model:cell(args.model,args.condition,args.batch,args.device)
    else:main(args.device)
