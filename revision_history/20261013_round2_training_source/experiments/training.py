"""Fresh three-way group split, full-state resume, existing shared train loop."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .common import (ROOT,CLASSES,ManifestDataset,build_model,infer,load_manifest,
                     prediction_csv,seed_all,sha256,status,transform,write_json)

sys.path.insert(0,str(ROOT/'train'))
from train_base import ModelEMA, build_param_groups, train_one_epoch
from evaluation_metrics import compute_metric_bundle


class ExternalDataset(Dataset):
    def __init__(self,images,labels,train,size):
        self.images,self.labels=images,labels.reshape(-1).astype(np.int64)
        self.transform=transform(train,size)
    def __len__(self): return len(self.labels)
    def __getitem__(self,i):
        return self.transform(Image.fromarray(self.images[i]).convert('RGB')),int(self.labels[i])


def download_external(name):
    import medmnist
    from medmnist.info import INFO
    info = INFO[name]
    root = ROOT/'results/external_data'
    root.mkdir(parents=True,exist_ok=True)
    cls = getattr(medmnist,info['python_class'])
    # Official 224-pixel release and its published checksum, never 28->224 silently.
    dataset = cls(root=str(root),split='train',download=True,size=224)
    path = root/(name+'_224.npz')
    md5 = hashlib.md5(path.read_bytes()).hexdigest()
    if md5 != info['MD5_224']: raise ValueError('External published MD5 mismatch')
    write_json(root/(name+'_provenance.json'),dict(
        source=info['url_224'],published_md5=info['MD5_224'],actual_md5=md5,
        license=info['license'],description=info['description'],
        official_split_sizes=info['n_samples'],resolution=224,
        labels=info['label'],patient_or_lesion_ids_in_npz=False,
        clinical_canine_external_validation=False))
    return np.load(path),[info['label'][str(i)] for i in range(len(info['label']))]


def seed_worker(_):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed);random.seed(worker_seed)


def train_one(args,name,seed,external=None):
    if args.phase=='pretrained' and name in ['psan','gfnet','gfnet_matched','afno']:
        raise ValueError(f'No ImageNet preset is loaded for {name}; use DeiT/ConvNeXt/FastViT/Swin pretrained baselines')
    cfg=json.loads(Path(args.config).read_text())
    cfg.update(model=name,seed=seed,device=args.device,dataset=args.dataset,
               atom=args.atom,atom_dropout=args.atom_dropout,atoms=args.atoms,
               init=args.init,axis_aligned=args.axis_aligned,fixed_phase=args.fixed_phase,
               pretrained=args.phase=='pretrained')
    if args.epochs is not None: cfg['epochs']=args.epochs
    cfg['source_code_sha256']={str(p.relative_to(ROOT)):sha256(p) for p in
        [ROOT/'experiments/training.py',ROOT/'experiments/common.py',ROOT/'models/psan.py',
         ROOT/'models/gfnet.py',ROOT/'models/afno.py',ROOT/'train/train_base.py']}
    if args.dataset=='canine':
        frame=load_manifest()
        cfg['split_sha256']=sha256(ROOT/'results/revision_split.csv')
        frames={s:frame[frame.split==s].reset_index(drop=True) for s in ['train','val','test']}
        datasets={s:ManifestDataset(f,s=='train',cfg['img_size']) for s,f in frames.items()}
        classes=CLASSES
    else:
        data,classes=external
        cfg['revision_protocol']=f'official_medmnist_{cfg["img_size"]}_'+args.dataset
        cfg['split_sha256']=sha256(ROOT/'results/external_data'/(args.dataset+'_provenance.json'))
        frames={s:pd.DataFrame({'path':[args.dataset+'/'+s+'/'+str(i) for i in range(len(data[s+'_labels']))],
                              'label':data[s+'_labels'].reshape(-1),
                              'group_id':[args.dataset+'/'+s+'/'+str(i) for i in range(len(data[s+'_labels']))]})
                for s in ['train','val','test']}
        datasets={s:ExternalDataset(data[s+'_images'],data[s+'_labels'],s=='train',cfg['img_size'])
                  for s in frames}
    cfg['num_classes']=len(classes)
    cfg['classes']=classes
    cfg['sample_counts']={s:len(d) for s,d in datasets.items()}
    extra = f'_{args.atom}_ad{str(args.atom_dropout).replace(".","p")}_M{args.atoms}_{args.init}' if name=='psan' else ''
    if args.axis_aligned: extra+='_axis'
    if args.fixed_phase: extra+='_fixedphase'
    if cfg['pretrained']: extra+='_imagenet'
    tag=f'{args.dataset}_{name}{extra}_s{seed}'
    run_dir=ROOT/'results/training'/tag
    run_dir.mkdir(parents=True,exist_ok=True)
    fingerprint=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
    config_path=run_dir/'config.json'
    result_path=run_dir/'summary.json'
    if result_path.exists() and config_path.exists():
        previous=json.loads(config_path.read_text())
        # Completed outputs retain their original source snapshot. Reusing them
        # does not execute changed code or rewrite their recorded configuration.
        ignored={'source_code_sha256','device','revision_protocol'}
        if {k:v for k,v in previous.items() if k not in ignored}!={k:v for k,v in cfg.items() if k not in ignored}:
            raise ValueError(f'Completed-run protocol mismatch for {tag}; preserve the run and use a new tag')
        summary=json.loads(result_path.read_text())
        if sha256(ROOT/summary['checkpoint'])!=summary['checkpoint_sha256']:
            raise ValueError('Completed checkpoint hash mismatch')
        if not all((run_dir/(s+'_predictions.csv')).exists() for s in ['val','test']):
            raise ValueError('Completed predictions missing')
        print('Reusing completed verified run',tag,flush=True)
        return summary
    if config_path.exists() and json.loads(config_path.read_text()) != cfg:
        raise ValueError(f'Configuration mismatch for existing run {tag}; use a new configuration/tag')
    write_json(config_path,cfg)
    if result_path.exists():
        print('Reusing completed verified run',tag,flush=True)
        return json.loads(result_path.read_text())
    status(tag,'IN PROGRESS',config=str(config_path.relative_to(ROOT)))
    seed_all(seed)
    device=torch.device(args.device)
    if device.type=='cuda': torch.cuda.set_device(device)
    model=build_model(name,classes=len(classes),size=cfg['img_size'],atom=args.atom,
                      dropout=args.atom_dropout,atoms=args.atoms,init=args.init,
                      axis_aligned=args.axis_aligned,fixed_phase=args.fixed_phase,
                      pretrained=cfg['pretrained']).to(device)
    no_decay=('norm','filter_real','filter_imag','mu','log_sigma','rho_raw','theta','log_amp','phi','w_real','w_imag')
    optimizer=torch.optim.AdamW(build_param_groups(model,cfg['weight_decay'],no_decay),lr=cfg['lr'])
    ema=ModelEMA(model,cfg['ema_decay'],device)
    labels=datasets['train'].labels
    counts=np.bincount(labels,minlength=len(classes))
    sampler_rng=torch.Generator().manual_seed(seed+1000)
    loader_rng=torch.Generator().manual_seed(seed+2000)
    sampler=WeightedRandomSampler(torch.as_tensor((1/np.sqrt(counts))[labels],dtype=torch.double),
                                  len(labels),replacement=True,generator=sampler_rng)
    loaders={}
    for split,dataset in datasets.items():
        loaders[split]=DataLoader(dataset,batch_size=cfg['batch_size'],
             sampler=sampler if split=='train' else None,shuffle=False,
             num_workers=cfg['num_workers'],pin_memory=device.type=='cuda',
             worker_init_fn=seed_worker,generator=loader_rng,persistent_workers=False)
    start=1;best=-1;best_epoch=0;bad=0;total_time=0;rows=[]
    last_path=run_dir/'last.pth';best_path=run_dir/'best.pth'
    if last_path.exists():
        if not args.resume: raise ValueError(f'{tag} is partial: explicitly use --resume')
        checkpoint=torch.load(last_path,map_location=device,weights_only=False)
        if checkpoint['config_fingerprint'] != fingerprint: raise ValueError('Resume config changed')
        model.load_state_dict(checkpoint['model']);ema.ema.load_state_dict(checkpoint['ema'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start=checkpoint['epoch']+1;best=checkpoint['best_macro_f1'];best_epoch=checkpoint['best_epoch']
        bad=checkpoint['patience_counter'];total_time=checkpoint['training_time_sec'];rows=checkpoint['history']
        rng=checkpoint['rng'];torch.set_rng_state(rng['torch'].cpu());np.random.set_state(rng['numpy'])
        random.setstate(rng['python']);sampler_rng.set_state(rng['sampler'].cpu());loader_rng.set_state(rng['loader'].cpu())
        if device.type=='cuda': torch.cuda.set_rng_state(rng['cuda'].cpu(),device)
        print('Resumed',tag,'at epoch',start,flush=True)
    print('START',tag,cfg,flush=True)
    for epoch in range(start,cfg['epochs']+1):
        if bad >= cfg['patience']: break
        tic=time.perf_counter()
        if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
        warmup=cfg['warmup_epochs']
        factor=epoch/warmup if epoch<=warmup else .5*(1+math.cos(math.pi*(epoch-warmup)/(cfg['epochs']-warmup)))
        for group in optimizer.param_groups:group['lr']=cfg['lr']*factor
        tr_loss,tr_acc,extra_log=train_one_epoch(model,loaders['train'],optimizer,device,ema)
        val_logits,val_y=infer(ema.ema,loaders['val'],device)
        val_f1=float(f1_score(val_y,val_logits.argmax(-1),labels=range(len(classes)),average='macro',zero_division=0))
        if device.type=='cuda':torch.cuda.synchronize(device)
        elapsed=time.perf_counter()-tic;total_time+=elapsed
        is_best=val_f1>best
        if is_best: best=val_f1;best_epoch=epoch;bad=0
        else:bad+=1
        rows.append(dict(epoch=epoch,train_loss=tr_loss,train_accuracy=tr_acc/100,
            validation_macro_f1=val_f1,learning_rate=cfg['lr']*factor,epoch_time_sec=elapsed,
            training_time_sec=total_time,gradient_norm_mean=extra_log['gradient_norm_mean'],
            peak_gpu_allocated_mb=torch.cuda.max_memory_allocated(device)/1024**2 if device.type=='cuda' else 0))
        checkpoint=dict(epoch=epoch,model=model.state_dict(),ema=ema.ema.state_dict(),
            eval_model=ema.ema.state_dict(),optimizer=optimizer.state_dict(),config=cfg,
            config_fingerprint=fingerprint,best_macro_f1=best,best_epoch=best_epoch,
            patience_counter=bad,training_time_sec=total_time,history=rows,
            rng=dict(torch=torch.get_rng_state(),numpy=np.random.get_state(),python=random.getstate(),
                sampler=sampler_rng.get_state(),loader=loader_rng.get_state(),
                cuda=torch.cuda.get_rng_state(device) if device.type=='cuda' else None))
        temporary=last_path.with_suffix('.tmp');torch.save(checkpoint,temporary);temporary.replace(last_path)
        if is_best:
            temporary=best_path.with_suffix('.tmp');torch.save(checkpoint,temporary);temporary.replace(best_path)
        pd.DataFrame(rows).to_csv(run_dir/'epochs.csv',index=False)
        print(f'{tag} epoch={epoch}/{cfg["epochs"]} val_F1={val_f1:.5f} best={best:.5f} time={elapsed:.1f}s',flush=True)
    checkpoint=torch.load(best_path,map_location=device,weights_only=False)
    model.load_state_dict(checkpoint['eval_model']);model.eval()
    evaluations={}
    for split in ['val','test']:
        logits,y=infer(model,loaders[split],device)
        prediction_csv(run_dir/(split+'_predictions.csv'),frames[split],logits,y,classes)
        bundle=compute_metric_bundle(y,torch.as_tensor(logits).softmax(-1).numpy(),classes)
        write_json(run_dir/(split+'_metrics.json'),bundle)
        evaluations[split]=bundle['metrics']
    summary=dict(tag=tag,model=name,dataset=args.dataset,seed=seed,best_validation_epoch=best_epoch,
        best_validation_macro_f1=best,epochs_completed=rows[-1]['epoch'],max_epochs=cfg['epochs'],
        training_time_sec=total_time,checkpoint=str(best_path.relative_to(ROOT)),
        checkpoint_sha256=sha256(best_path),config_fingerprint=fingerprint,
        split_sha256=cfg['split_sha256'],parameters=sum(p.numel() for p in model.parameters()),
        pretrained=cfg['pretrained'],atom=args.atom if name=='psan' else '',
        atom_dropout=args.atom_dropout if name=='psan' else None,
        peak_gpu_allocated_mb=max(r['peak_gpu_allocated_mb'] for r in rows),**evaluations)
    write_json(result_path,summary);status(tag,'COMPLETED',summary=str(result_path.relative_to(ROOT)))
    print('COMPLETED',tag,summary,flush=True)
    return summary


def run(args):
    torch.set_num_threads(4)
    if not args.train:
        from .analysis import aggregate_new_runs
        aggregate_new_runs(args)
        return
    external=None
    if args.dataset!='canine': external=download_external(args.dataset)
    for name in args.models:
        for seed in args.seeds:
            train_one(args,name,seed,external)
    from .analysis import aggregate_new_runs
    aggregate_new_runs(args)
