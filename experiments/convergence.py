"""Empirical finite-budget learning curves; no AdamW convergence theorem."""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analysis import savefig
from .common import ROOT,sha256,write_json


def run():
    records=[];diagnostics=[];sources={}
    fig,axes=plt.subplots(1,2,figsize=(9,3.4))
    for model,color in [('psan','#4477aa'),('gfnet','#228833'),('afno','#aa3377')]:
        curves=[]
        for seed in [42,43,44,45,46]:
            suffix='_gabor_ad0p2_M16_morlet' if model=='psan' else ''
            directory=ROOT/'results/training'/f'canine_{model}{suffix}_s{seed}'
            if not (directory/'summary.json').exists():continue
            p=directory/'epochs.csv';frame=pd.read_csv(p);curves.append(frame)
            sources[str(p.relative_to(ROOT))]=sha256(p)
            tail=frame.tail(5)
            diagnostics.append(dict(model=model,seed=seed,epochs=len(frame),
                last_validation_f1=float(frame.validation_macro_f1.iloc[-1]),
                last_five_validation_f1_change=float(tail.validation_macro_f1.iloc[-1]-tail.validation_macro_f1.iloc[0]),
                last_five_validation_f1_slope=float(np.polyfit(tail.epoch,tail.validation_macro_f1,1)[0]),
                last_train_loss=float(frame.train_loss.iloc[-1])))
        data=pd.concat(curves,ignore_index=True)
        for epoch,group in data.groupby('epoch'):
            records.append(dict(model=model,epoch=int(epoch),runs=len(group),
                mean_train_loss=float(group.train_loss.mean()),sd_train_loss=float(group.train_loss.std(ddof=1)),
                mean_validation_macro_f1=float(group.validation_macro_f1.mean()),
                sd_validation_macro_f1=float(group.validation_macro_f1.std(ddof=1))))
        subset=pd.DataFrame(records);subset=subset[subset.model==model]
        x=subset.epoch.to_numpy()
        for ax,mean,sd in [(axes[0],'mean_train_loss','sd_train_loss'),
                           (axes[1],'mean_validation_macro_f1','sd_validation_macro_f1')]:
            y=subset[mean].to_numpy();s=subset[sd].to_numpy()
            ax.plot(x,y,label=f'{model.upper()} (n={len(curves)})',color=color)
            ax.fill_between(x,y-s,y+s,color=color,alpha=.12)
    axes[0].set(xlabel='Epoch',ylabel='Mean weighted-sampler training loss')
    axes[1].set(xlabel='Epoch',ylabel='Validation Macro F1')
    for ax in axes:ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.tight_layout();savefig(fig,ROOT/'results/figures/convergence')
    pd.DataFrame(records).to_csv(ROOT/'results/tables/convergence_curves.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(ROOT/'results/tables/convergence_diagnostics.csv',index=False)
    write_json(ROOT/'results/convergence_provenance.json',dict(source_sha256=sources,
        source_code_sha256=sha256(__file__),interpretation='mean plus/minus sample SD of actual epoch records; finite30epoch schedule',
        guarantee='No global convergence guarantee for the nonlinear atom/network/AdamW optimization is claimed',
        test_used=False))


if __name__=='__main__':run()
