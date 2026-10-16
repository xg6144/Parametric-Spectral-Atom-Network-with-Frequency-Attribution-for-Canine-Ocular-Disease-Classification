"""Requested controlled OFAT; non-filter initialization matches M16 seed donor."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from . import training
from .common import ROOT,build_model,write_json


GROUPS={0:[dict(atoms=4),dict(atoms=8)],
        1:[dict(atoms=32),dict(atoms=64)],
        2:[dict(init='random'),dict(axis_aligned=True)],
        3:[dict(fixed_phase=True),dict(atom_dropout=.1)]}


def main():
    p=argparse.ArgumentParser();p.add_argument('--queue',type=int,required=True);p.add_argument('--device',required=True)
    args=p.parse_args();torch.set_num_threads(4)
    args.phase='multiseed';args.dataset='canine';args.epochs=None;args.resume=True
    args.models=['psan'];args.seeds=[42];args.seed=42;args.train=True
    baseline=json.loads((ROOT/'configs/revision.json').read_text())
    baseline['initialization_protocol']='Non-filter tensors match canonical untrained PSAN M16/seed42; M16 mixing weights also match. Atom geometry alone changes for random init.'
    baseline['initializer_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    config=ROOT/'configs/ofat_matched_initialization.json'
    if config.exists() and json.loads(config.read_text())!=baseline:raise ValueError('Frozen OFAT config differs')
    if not config.exists():write_json(config,baseline)
    args.config=str(config)
    original_builder=training.build_model
    def matched_builder(name,**kwargs):
        model=original_builder(name,**kwargs)
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(torch.Generator().manual_seed(args.seed).get_state())
            donor=build_model('psan',classes=kwargs['classes'],size=kwargs['size'])
        source=donor.state_dict();target=model.state_dict()
        copied=[]
        with torch.no_grad():
            for key,value in target.items():
                non_filter='.filter.' not in key
                mixing=args.atoms==16 and key.endswith(('.w_real','.w_imag'))
                if (non_filter or mixing) and key in source and value.shape==source[key].shape:
                    value.copy_(source[key]);copied.append(key)
        model.load_state_dict(target)
        print('Controlled initial tensors copied',len(copied),'atoms',args.atoms,'init',args.init,flush=True)
        return model
    training.build_model=matched_builder
    for overrides in GROUPS[args.queue]:
        args.atom='gabor';args.atom_dropout=0.;args.atoms=16;args.init='morlet';args.axis_aligned=False;args.fixed_phase=False
        for key,value in overrides.items():setattr(args,key,value)
        training.train_one(args,'psan',42)
    from .analysis import aggregate_new_runs
    aggregate_new_runs(args)


if __name__=='__main__':main()
