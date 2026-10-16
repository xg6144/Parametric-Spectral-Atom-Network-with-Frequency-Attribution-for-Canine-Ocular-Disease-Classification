"""Fixed follow-up replications. No existing run is overwritten."""
import argparse
import json
from types import SimpleNamespace

import torch

from .common import ROOT,sha256,status,write_json
from .external import prepare
from .training import train_one


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker',type=int,choices=[0,1],required=True)
    parser.add_argument('--device',required=True);args=parser.parse_args()
    torch.set_num_threads(4)
    seed=45+args.worker;factor_seed=43+args.worker
    cfg=SimpleNamespace(phase='multiseed',dataset='canine',config=str(ROOT/'configs/revision.json'),
        epochs=None,resume=True,device=args.device,atom='gabor',atom_dropout=.2,atoms=16,
        init='morlet',axis_aligned=False,fixed_phase=False)
    tasks=[('core',name,seed,'gabor',.2) for name in ['psan','gfnet','afno']]
    tasks += [('factorial','psan',factor_seed,atom,ad) for atom,ad in [('gabor',0.),('gaussian',0.),('gaussian',.2)]]
    tasks += [('matched','gfnet_matched',factor_seed,'gabor',.2)]
    for scope,name,seed,atom,ad in tasks:
        cfg.phase='multiseed';cfg.dataset='canine';cfg.config=str(ROOT/'configs/revision.json')
        cfg.atom=atom;cfg.atom_dropout=ad
        print('ROUND2 task',scope,name,seed,atom,ad,flush=True)
        train_one(cfg,name,seed)
    cfg.phase='external';cfg.atom='gabor';cfg.atom_dropout=.2
    cfg.config=str(ROOT/'configs/external64.json');cfg.dataset='octmnist64'
    data=prepare('octmnist',64)
    for name in ['psan','gfnet','afno']:train_one(cfg,name,factor_seed,data)
    if args.worker==0:
        cfg.phase='pretrained';cfg.dataset='canine';cfg.config=str(ROOT/'configs/revision.json')
        train_one(cfg,'shvit_s1',42)
    else:
        cfg.phase='external';cfg.dataset='bloodmnist64';cfg.config=str(ROOT/'configs/external64.json')
        data=prepare('bloodmnist',64)
        for seed in [42,43,44]:
            for name in ['psan','gfnet','afno']:train_one(cfg,name,seed,data)
    status('round2_worker'+str(args.worker),'COMPLETED')


if __name__=='__main__':main()
