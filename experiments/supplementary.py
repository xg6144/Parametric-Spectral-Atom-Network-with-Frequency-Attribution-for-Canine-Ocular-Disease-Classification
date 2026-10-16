"""CPU-only resource and rank/performance measurements from actual checkpoints."""
from __future__ import annotations

import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import recall_score

from model_resources import measure_supported_operations
from .analysis import read_new,savefig
from .common import ROOT,build_model,load_model,sha256,status,write_json


def resources():
    torch.set_num_threads(4)
    rows=[];details={}
    for name in ['psan','gfnet','gfnet_matched','afno']:
        torch.manual_seed(42)
        model=build_model(name).eval()
        fv=measure_supported_operations(model,torch.device('cpu'),224)
        with torch.no_grad(),torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU],with_flops=True) as profiler:
            model(torch.zeros(1,3,224,224))
        events={e.key:int(e.flops) for e in profiler.key_averages() if e.flops}
        details[name]=dict(fvcore=fv,torch_profiler_events=events,
            input_shape=[1,3,224,224],precision='float32',cache=False,
            limitation='supported operators only; complete FFT/complex/transcendental costs absent')
        rows.append(dict(model=name,parameters=sum(p.numel() for p in model.parameters()),
            fvcore_supported_arithmetic_operations=fv['flops'] if fv else np.nan,
            fvcore_supported_macs=fv['macs'] if fv else np.nan,
            torch_supported_arithmetic_operations=sum(events.values()),complete_flop_count=False))
    pd.DataFrame(rows).to_csv(ROOT/'results/tables/model_operations.csv',index=False)
    write_json(ROOT/'results/model_operations.json',dict(pytorch=torch.__version__,models=details,
        fvcore_convention='known FMA=2 scalar arithmetic operations; never divide fvcore FMA count by2',
        source_code_sha256=sha256(__file__)))


def rank_performance():
    torch.set_num_threads(4)
    rows=[]
    for m in [4,8,16,32,64]:
        path=ROOT/'results/training'/f'canine_psan_gabor_ad0p0_M{m}_morlet_s42'
        summary=json.loads((path/'summary.json').read_text())
        model,_,_=load_model(path/'best.pth','cpu')
        with torch.no_grad():
            for index,block in enumerate(model.blocks):
                k=block.filter.build_filter().reshape(192,-1)
                singular=torch.linalg.svdvals(k).double().numpy()
                p=singular**2/np.square(singular).sum()
                entropy_rank=float(np.exp(-(p[p>0]*np.log(p[p>0])).sum()))
                rows.append(dict(atoms=m,block=index,
                    nominal_numerical_rank=int((singular>singular[0]*1e-6).sum()),
                    entropy_effective_rank=entropy_rank,
                    tail_energy_after_M=float(np.square(singular[m:]).sum()/np.square(singular).sum()),
                    macro_f1=summary['test']['macro_f1'],validation_macro_f1=summary['val']['macro_f1'],
                    checkpoint=summary['checkpoint'],checkpoint_sha256=summary['checkpoint_sha256']))
    df=pd.DataFrame(rows);df.to_csv(ROOT/'results/tables/rank_performance.csv',index=False)
    mean=df.groupby('atoms').mean(numeric_only=True)
    fig,axes=plt.subplots(1,2,figsize=(8.5,3.1))
    axes[0].plot(mean.index,mean.macro_f1,'o-',color='#4477aa')
    axes[0].set(xlabel='Atoms M',ylabel='Test Macro F1',title='One controlled training seed')
    axes[1].plot(mean.index,mean.entropy_effective_rank,'s-',color='#228833',label='Entropy effective rank')
    axes[1].plot(mean.index,mean.nominal_numerical_rank,'o--',color='#aa3377',label='Numerical rank')
    axes[1].set(xlabel='Atoms M',ylabel='Mean nominal filter rank',title='Across 12 trained blocks')
    for ax in axes:
        ax.set_xscale('log',base=2);ax.set_xticks([4,8,16,32,64],labels=['4','8','16','32','64'])
    axes[1].legend(fontsize=8);fig.tight_layout();savefig(fig,ROOT/'results/figures/rank_performance')
    write_json(ROOT/'results/rank_performance_protocol.json',dict(seed=42,atom_dropout=0,
        thresholds='numerical rank: singular value > largest*1e-6; entropy rank from squared singular energies',
        scope='descriptive relation; not causal rank regularization inference; no test-directed model selection',
        source_code_sha256=sha256(__file__)))


def device_recall():
    path=ROOT/'results/tables/device_performance.csv'
    df=pd.read_csv(path,keep_default_na=False)
    for tag,part in df.groupby('tag'):
        frame,prob,_,names=read_new(ROOT/'results/training'/tag/'test_predictions.csv')
        for index,row in part.iterrows():
            classes=[names.index(c) for c in row.supported_classes.split('|')]
            f=frame[frame.device==row.device]
            if row.protocol=='common_true_classes_full_classifier':f=f[f.label.isin(classes)]
            df.loc[index,'macro_recall_supported']=recall_score(f.label,prob[f.index].argmax(1),labels=classes,average='macro',zero_division=0)
    df.to_csv(path,index=False)


def run(args):
    status('supplementary','IN PROGRESS')
    resources();rank_performance();device_recall()
    status('supplementary','COMPLETED',profiles=4,rank_blocks=60)


if __name__=='__main__':run(None)
