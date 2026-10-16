from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from models.psan import PSANConfig,PSANFilter
from .common import ROOT,load_model,sha256,status,write_json
from .analysis import savefig


def run(args):
    status('theory','IN PROGRESS');torch.set_num_threads(4)
    device=torch.device(args.device)
    checkpoint=args.checkpoint or next((ROOT/'train_results/sgkf5_fs42_psan_best_fold1_seed42').glob('*/best*.pth'))
    model,_,_=load_model(checkpoint,device)
    rows=[];spectra=[]
    with torch.no_grad():
        for i,block in enumerate(model.blocks):
            f=block.filter;k=f.build_filter().reshape(f.cfg.dim,-1)
            singular=torch.linalg.svdvals(k).cpu().numpy()
            spectra.append(singular)
            pr,pi=f._atoms_complex();psi=torch.complex(pr,pi).reshape(f.cfg.M,-1)
            norms=psi.norm(dim=1);normalized=psi/norms[:,None].clamp(min=1e-12)
            gram=(normalized@normalized.conj().T).abs();gram.fill_diagonal_(0)
            svp=torch.linalg.svdvals(psi).cpu().numpy()
            w=torch.complex(f.w_real,f.w_imag)
            crude=(w.abs()*f.log_amp.exp()[None]).sum(1).max()
            rows.append(dict(block=i,atoms=f.cfg.M,nominal_rank_at_relative_1e6=int((singular>singular[0]*1e-6).sum()),
                singular_tail_energy_after_M=float((singular[f.cfg.M:]**2).sum()/(singular**2).sum()),
                atom_max_coherence=float(gram.max()),atom_basis_condition=float(svp[0]/max(svp[-1],1e-30)),
                max_nominal_filter_gain=float(k.abs().max()),coefficient_amplitude_bound=float(crude)))
    pd.DataFrame(rows).to_csv(ROOT/'results/tables/psan_filter_structure.csv',index=False)
    np.savez(ROOT/'results/psan_filter_singular_values.npz',singular_values=np.asarray(spectra))
    fig,ax=plt.subplots(figsize=(6.5,4))
    for i,s in enumerate(spectra):ax.semilogy(np.arange(1,len(s)+1),np.maximum(s/s[0],1e-10),alpha=.6,label='Block '+str(i))
    ax.axvline(16,color='black',ls='--');ax.set(xlabel='Singular value index',ylabel='Singular value / largest',title='Nominal channel × frequency filter spectrum');ax.legend(ncol=3,fontsize=7);savefig(fig,ROOT/'results/figures/filter_svd')
    gf_checkpoint=next((ROOT/'train_results/gfnet_tiny').glob('*/best*.pth'))
    ckpt=torch.load(gf_checkpoint,map_location='cpu',weights_only=False)
    state=ckpt[next(k for k in ['eval_model','ema','model'] if ckpt.get(k) is not None)]
    from .full_basis import capacity_check
    full_basis=capacity_check(state)
    approximation=[]
    # Fitting a filter tensor is not network retraining and uses no test labels.
    for i in range(12):
        # Repository GFNet calls the token mixer 'mixer', unlike PSAN's 'filter'.
        target=torch.complex(state[f'blocks.{i}.mixer.filter_real'],state[f'blocks.{i}.mixer.filter_imag']).permute(2,0,1).to(device)
        d,h,wh=target.shape;w=(wh-1)*2
        s=torch.linalg.svdvals(target.reshape(d,-1)).detach()
        energy=s.square().sum()
        for m in [4,8,16,32,64]:
            torch.manual_seed(args.seed+i*100+m)
            fit=PSANFilter(PSANConfig(d,h,w,M=m,atom='gaussian',atom_dropout=0)).to(device).eval()
            # Start with the best ridge-stabilized mixing for the initial geometry.
            with torch.no_grad():
                re,im=fit._atoms_complex();psi=torch.complex(re,im).reshape(m,-1)
                mix=target.reshape(d,-1)@torch.linalg.pinv(psi,rcond=1e-6)
                fit.w_real.copy_(mix.real);fit.w_imag.copy_(mix.imag)
            optimizer=torch.optim.Adam(fit.parameters(),lr=.01)
            best=float('inf');best_state=None;initial=None
            for step in range(201):
                estimate=fit.build_filter();loss=(estimate-target).abs().square().sum()/energy
                val=float(loss.detach())
                if initial is None:initial=val
                if val<best:best=val;best_state={k:v.detach().cpu().clone() for k,v in fit.state_dict().items()}
                if step==200:break
                optimizer.zero_grad();loss.backward();optimizer.step()
                with torch.no_grad():fit.log_sigma.clamp_(-5,2);fit.log_amp.clamp_(-5,5)
            lower=float(s[m:].square().sum()/energy)
            row=dict(block=i,atoms=m,relative_frobenius_error_initial=np.sqrt(initial),
                relative_frobenius_error_fitted=np.sqrt(best),unconstrained_rank_M_svd_lower_bound=np.sqrt(lower),
                optimization_steps=200,atom='gaussian',seed=args.seed+i*100+m)
            approximation.append(row)
            directory=ROOT/'results/filter_approximation';directory.mkdir(exist_ok=True)
            torch.save(best_state,directory/f'gfnet_block{i}_M{m}.pth')
            print('Approximation',row,flush=True)
    pd.DataFrame(approximation).to_csv(ROOT/'results/tables/gfnet_atom_approximation.csv',index=False)
    table=pd.DataFrame(approximation).groupby('atoms')[['relative_frobenius_error_fitted','unconstrained_rank_M_svd_lower_bound']].mean()
    fig,ax=plt.subplots(figsize=(6.5,4));ax.plot(table.index,table.relative_frobenius_error_fitted,'o-',label='Fitted Gaussian atoms');ax.plot(table.index,table.unconstrained_rank_M_svd_lower_bound,'s--',label='Unrestricted rank-M SVD bound');ax.set(xlabel='Number of atoms M',ylabel='Relative Frobenius error (mean across 12 blocks)',xscale='log');ax.set_xticks([4,8,16,32,64],labels=['4','8','16','32','64']);ax.legend();savefig(fig,ROOT/'results/figures/filter_approximation')
    write_json(ROOT/'results/theory_provenance.json',dict(psan_checkpoint=str(Path(checkpoint).relative_to(ROOT)),
        psan_sha256=sha256(checkpoint),gfnet_checkpoint=str(gf_checkpoint.relative_to(ROOT)),gfnet_sha256=sha256(gf_checkpoint),
        fitting_uses_test_labels=False,steps_per_fit=200,optimizer='Adam lr=.01',
        log_sigma_clamp=[-5,2],log_amplitude_clamp=[-5,5],fits=60,
        full_basis_finite_grid_capacity=full_basis))
    (ROOT/'reports/09_theoretical_analysis.md').write_text(r'''# Structured representation, approximation and stability

Status: COMPLETED for mathematical analysis and executed trained-filter SVD/approximation; rank–classification-performance causality is not established.

Flatten F=H(floor(W/2)+1) frequency bins. The nominal learned half-spectrum matrix satisfies K=WΨ, W∈C^(D×M), Ψ∈C^(M×F), hence rank(K)≤min(D,M,F). This is a shared channel–frequency span constraint, stronger than parameter counting alone. GFNet learns arbitrary nominal entries in C^(D×F). A nonlinear AFNO mixer is not the same linear multiplier. Parameter reduction does not by itself prove low rank for any other architecture or improved classification. The nominal PSAN rank constraint does not imply that an unconstrained low-rank approximation is itself an admissible parametric Gaussian atom bank.

For target A with singular values s_j, any rank-M approximation has squared Frobenius residual at least sum_(j>M) s_j²; truncated SVD attains this among arbitrary rank-M matrices. Parametric atom fits are a constrained subset, so cannot beat the bound except for rounding error. We actually fitted all 12 stored GFNet filters for M4/8/16/32/64 (200 Adam steps, Gaussian geometry and complex weights; 60 fits). Best visited fit is saved; nonconvex optimization need not find the optimum. These tensor fits do not use images/test labels and do not establish preserved network accuracy. An increase in M expands capacity when smaller banks can be embedded by zeroing added coefficients, but independent finite-budget fits need not decrease monotonically.

On a fixed finite grid with F distinct points, an ideal Gaussian bank with M=F, centers at those points and sufficiently small positive covariance has a basis matrix approaching a nonzero diagonal matrix. By continuity its determinant is nonzero for sufficiently small widths, so arbitrary nominal finite-grid filters can be represented by suitable complex W. This constructive statement concerns the ideal parameterization (or the axis-aligned implementation without determinant clipping); it is not a universal-approximation theorem for a fixed small M, the numerical full-covariance determinant floor, bounded fitting optimizer, or the nonlinear network. M16≪F112 imposes a real capacity restriction. Phase exp(iφ_m) can be absorbed into unconstrained W[:,m]; trainable constant phase cannot enlarge the function class.

For a full orthonormal Fourier transform and bounded multiplier K, Parseval gives ||T_K x||₂²=sum_(frequency,channel)|K|²|Fx|²≤||K||∞²||x||₂². Hence ||T_K||_(2→2)≤||K||∞≤max_d sum_m |w_dm| A_m when Gaussian envelopes lie in [0,A_m]. This assumes finite positive scales/amplitudes and the fixed trained filter. Learned amplitudes/weights are unbounded a priori, so it is a conditional bound, not a uniform training stability guarantee. Residuals, LayerNorm (dependent on epsilon/scales), MLPs and compositions require separate Lipschitz bounds. We do not assert whole-network noise robustness from this filter inequality.

For real rFFT/irFFT, boundary columns v=0 and the even-width Nyquist column must be Hermitian along u. Arbitrary nominal K does not enforce this. The actual real output implicitly projects the boundary product to Hermitian symmetry, with effective boundary multiplier (K(u,v)+conj(K(-u,v)))/2. This projection is contractive in the correctly weighted Parseval norm, so the same max-gain bound holds. However nominal matrix rank≤M does not automatically hold for the Hermitian-projected effective response: conjugate mixing may require up to 2M basis vectors. This distinction is explicit; no extra projection is silently inserted into historical models. Filtered Spectral-CAM uses rFFT of the actual irFFT output.

Dropout uses B_m~Bernoulli(1-p), S=sum B, mask_m=M B_m/S for S>0 and zero for S=0. Symmetry gives E(mask_m)=P(S>0)=1-p^M. Conditional on S>0 it is unbiased; unconditionally it is slightly attenuated. For p=.2,M16, p^M=6.5536e-12; for smaller M the exception is larger. A shared realized mask preserves exact per-atom branch decomposition; independently sampled masks do not. The decomposition and gradient inner product do not imply final-logit completeness through nonlinear suffix layers.

Actual tables: `psan_filter_structure.csv`, `gfnet_atom_approximation.csv`. Saved singular values and fitted states regenerate the vector plots `filter_svd.pdf` and `filter_approximation.pdf`. Conditional gain/coherence/condition measurements describe the analyzed stored checkpoint; its selected-validation classification score is not used as unbiased performance evidence.
''')
    status('theory','COMPLETED',trained_psan_blocks=12,gfnet_fits=60)
