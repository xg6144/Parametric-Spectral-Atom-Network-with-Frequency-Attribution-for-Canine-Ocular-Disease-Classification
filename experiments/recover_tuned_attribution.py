"""Validate completed numerical artifacts after a final metadata-path failure."""
import json

import numpy as np
import pandas as pd

from .analysis import read_new
from .attribution_diagnostics import run as additional
from .common import ROOT,load_manifest,sha256,status,write_json
from .faithfulness import mean_ci,safe_spearman


def run():
    checkpoint=ROOT/'results/tuning/final/canine_psan_c2_s42_final/best.pth'
    digest=sha256(checkpoint);directory=ROOT/'results/attribution'/digest[:12]
    frame=pd.read_csv(directory/'sample_manifest.csv',keep_default_na=False)
    expected=load_manifest();expected=expected[expected.split=='test'].groupby('label',group_keys=False).sample(n=20,random_state=42).sort_values('path').reset_index(drop=True)
    if not frame.equals(expected):raise ValueError('Saved attribution sample/protocol differs')
    values=json.loads((directory/'atom_values.json').read_text())
    if len(values)!=660:raise ValueError('Incomplete single-atom numerical results')
    for row in values:
        if len(row['attribution'])!=16 or len(row['actual_logit_drops'])!=16:raise ValueError('Incomplete atom budget')
        rho=safe_spearman(np.asarray(row['attribution']),np.asarray(row['actual_logit_drops']))
        if not np.isclose(rho,row['spearman'],equal_nan=True):raise ValueError('Spearman mismatch')
    native=pd.read_csv(directory/'native_mask_equivalence.csv')
    if len(native)!=1344 or native.max_abs_difference.max()>1e-5:raise ValueError('Native mask checks incomplete/failed')
    curves=pd.read_csv(directory/'deletion_curves.csv',keep_default_na=False)
    if len(curves)!=42240 or set(curves.k)!=set(range(1,17)):raise ValueError('Incomplete deletion budgets')
    sanity=pd.read_csv(directory/'sanity.csv',keep_default_na=False)
    spectral=pd.read_csv(directory/'spectral_deletion.csv',keep_default_na=False)
    if len(sanity)!=1320 or len(spectral)!=3520:raise ValueError('Incomplete sanity/frequency interventions')
    _,probs,_,_=read_new(checkpoint.with_name('test_predictions.csv'))
    # Hash and per-path target checks are repeated by the additional diagnostics.
    provenance=dict(checkpoint=str(checkpoint.relative_to(ROOT)),checkpoint_sha256=digest,
        active_split_sha256=sha256(ROOT/'results/revision_split.csv'),images=220,samples_per_class=20,
        blocks=[0,5,11],target='original predicted class fixed throughout',
        atom_removal='one block only; exactly zero removed atoms; no survivor rescaling; all other blocks unchanged',
        aopc='mean logit drop over k1..16 per image, then mean over the balanced diagnostic subset',
        spectral_deletion='actual real filter response rFFT; conjugate-consistent groups;1/10/25/50percent budgets',
        full_randomization='non-geometric parameters reset; geometry retained for index comparability',
        noise_sigma_rgb=.01,bootstrap=1000,cohort='equal_budget_validation_only_followup_v1',
        scope='validation-selected model; balanced model-behavior diagnostics, not expert clinical validation',
        executed_source_sha256='e0630256532c6f4338ce07b2967b5e3d602a7ab8ad1a15649df1f1951613620c',
        executed_source_snapshot='revision_history/new_source_versions/faithfulness_e0630256532c6f4338ce07b2967b5e3d602a7ab8ad1a15649df1f1951613620c.py',
        finalization_error='relative CLI checkpoint path failed relative_to(ROOT) after all raw calculations/figures were saved',
        failure_log='results/experiment_logs/tuned_attribution.log',
        recovery='validated complete saved atom/deletion/sanity/frequency outputs; no invented or repeated model measurements',
        baseline_numerics='replayed batch8 baseline matches220/220 fixed attribution targets; TF32-enabled convolution differs from batch128 at one case; controlled TF32-off results are archived',
        recovery_source_sha256=sha256(__file__))
    write_json(directory/'provenance.json',provenance)
    additional(directory)
    status('attribution','COMPLETED',provenance=provenance,summary=str((directory/'summary.csv').relative_to(ROOT)),
           recovered_after_metadata_error=True)
    print('Completed tuned attribution artifacts validated and finalized',flush=True)


if __name__=='__main__':run()
