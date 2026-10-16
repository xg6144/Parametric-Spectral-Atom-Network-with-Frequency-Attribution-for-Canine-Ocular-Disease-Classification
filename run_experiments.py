"""Unified revision interface, extending existing PSAN training/evaluation code."""
from __future__ import annotations

import argparse
import copy
import importlib
import traceback


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--phase', required=True, choices=['audit','calibration','ablation','multiseed',
                   'attribution','external','robustness','efficiency','theory','pretrained','tuning','cv','classwise','statistics','supplementary','finalize','all'])
    p.add_argument('--config', default='configs/revision.json')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--models', nargs='+')
    p.add_argument('--seeds', nargs='+', type=int, default=[42,43,44])
    p.add_argument('--checkpoint')
    p.add_argument('--dataset', default='canine')
    p.add_argument('--epochs', type=int)
    p.add_argument('--bootstrap', type=int, default=1000)
    p.add_argument('--samples-per-class', type=int, default=20)
    p.add_argument('--train', action='store_true', help='Actually run training for requested phase')
    p.add_argument('--resume', action='store_true', help='Resume verified checkpoint including optimizer/RNG')
    p.add_argument('--atom', choices=['gabor','gaussian'], default='gabor')
    p.add_argument('--atom-dropout', type=float, default=.2)
    p.add_argument('--atoms', type=int, default=16)
    p.add_argument('--init', choices=['morlet','random'], default='morlet')
    p.add_argument('--axis-aligned', action='store_true')
    p.add_argument('--fixed-phase', action='store_true')
    p.add_argument('--tuning-worker',type=int,choices=[0,1],help='Frozen tuning worker; omit to run both sequentially')
    p.add_argument('--cv-fold',type=int,choices=[2,3,4,5],help='Additional outer CV fold; fold1 reuses verified main runs')
    args = p.parse_args()
    if args.models is None:
        args.models=['deit','convnext','fastvit_t8','swin','shvit_s1'] if args.phase=='pretrained' else ['psan','gfnet','afno']
    mapping = {'audit':'audit', 'calibration':'analysis', 'ablation':'analysis',
               'multiseed':'training','external':'external','pretrained':'training',
               'attribution':'faithfulness','robustness':'robustness', 'efficiency':'efficiency',
               'theory':'theory','tuning':'tuning_results','cv':'outer_cv','classwise':'analysis','statistics':'statistics',
               'supplementary':'supplementary','finalize':'documents'}
    phases = list(mapping) if args.phase == 'all' else [args.phase]
    for phase in phases:
        phase_args=copy.deepcopy(args)
        phase_args.phase = phase
        try:
            if phase=='external' and args.phase=='all' and args.train:
                phase_args.config='configs/external64.json'
                for dataset in ['dermamnist_c64','octmnist64','bloodmnist64']:
                    external_args=copy.deepcopy(phase_args);external_args.dataset=dataset
                    importlib.import_module('experiments.external').run(external_args)
            else:
                if phase=='pretrained' and args.phase=='all':
                    phase_args.models=['deit','convnext','fastvit_t8','swin','shvit_s1']
                    phase_args.seeds=[42]
                importlib.import_module('experiments.' + mapping[phase]).run(phase_args)
        except Exception as exc:
            from experiments.common import ROOT, status
            trace = traceback.format_exc()
            (ROOT/'results/experiment_logs'/f'{phase}_error.log').write_text(trace)
            status(phase,'BLOCKED',error=str(exc),traceback=f'results/experiment_logs/{phase}_error.log')
            print(trace,flush=True)
            if len(phases) == 1: raise


if __name__ == '__main__':
    main()
