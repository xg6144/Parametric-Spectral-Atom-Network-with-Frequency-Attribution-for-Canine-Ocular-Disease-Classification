"""Executed external learning curves and stopping diagnostics, all seeds retained."""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from .analysis import savefig
from .common import ROOT,sha256,write_json


def run():
    datasets=['dermamnist_c64','octmnist64','bloodmnist64'];models=['psan','gfnet','afno']
    colors=['#4477aa','#ee6677','#228833'];rows=[];inputs=[]
    fig,axes=plt.subplots(3,3,figsize=(11,7.8),sharex=True,sharey=True)
    titles={'dermamnist_c64':'Corrected Derma64','octmnist64':'OCT64','bloodmnist64':'Blood64'}
    for i,dataset in enumerate(datasets):
        for j,model in enumerate(models):
            ax=axes[i,j]
            for k,seed in enumerate([42,43,44]):
                suffix='_gabor_ad0p2_M16_morlet' if model=='psan' else ''
                root=ROOT/'results/training'/f'{dataset}_{model}{suffix}_s{seed}'
                summary=json.loads((root/'summary.json').read_text());epochs=pd.read_csv(root/'epochs.csv')
                if len(epochs)!=summary['epochs_completed']:raise ValueError('External epoch log mismatch')
                ax.plot(epochs.epoch,epochs.validation_macro_f1,color=colors[k],label=f'Seed {seed}',linewidth=1.3)
                rows.append(dict(dataset=dataset,model=model,seed=seed,epochs_completed=len(epochs),
                    maximum_epochs=summary['max_epochs'],stopped_early=len(epochs)<summary['max_epochs'],
                    best_validation_epoch=summary['best_validation_epoch'],validation_macro_f1=summary['val']['macro_f1'],
                    test_macro_f1=summary['test']['macro_f1'],training_time_sec=summary['training_time_sec'],
                    checkpoint_sha256=summary['checkpoint_sha256']))
                inputs.append(dict(path=str((root/'epochs.csv').relative_to(ROOT)),sha256=sha256(root/'epochs.csv')))
            ax.set(title=f'{titles[dataset]} / {model.upper()}',xlim=(1,30),ylim=(0,1))
            if j==0:ax.set_ylabel('Validation Macro F1')
            if i==2:ax.set_xlabel('Training epoch')
            ax.grid(alpha=.2);ax.legend(fontsize=7,loc='lower right')
    fig.tight_layout();savefig(fig,ROOT/'results/figures/external_validation_curves')
    pd.DataFrame(rows).to_csv(ROOT/'results/tables/external_stopping_diagnostics.csv',index=False)
    write_json(ROOT/'results/external_stopping_protocol.json',dict(inputs=inputs,
        source_code_sha256=sha256(__file__),selection='unchanged validation-only early stopping; test scores displayed descriptively only',
        interpretation='seed42 Blood AFNO stopped early under the shared preset; adverse result retained, no selective retry or test-directed schedule change'))
    print('All27 external stopping records and nine-panel validation curves saved',flush=True)


if __name__=='__main__':run()
