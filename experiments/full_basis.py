import numpy as np
import pandas as pd
import torch

from models.psan import PSANFilter,PSANConfig
from .common import ROOT


def capacity_check(state):
    rows=[]
    for i in range(12):
        target=torch.complex(state[f'blocks.{i}.mixer.filter_real'],state[f'blocks.{i}.mixer.filter_imag']).permute(2,0,1).to(torch.complex128).cpu()
        d,h,wh=target.shape;w=(wh-1)*2;m=h*wh
        f=PSANFilter(PSANConfig(d,h,w,M=m,atom='gaussian',atom_dropout=0)).double().eval()
        with torch.no_grad():
            f.mu.copy_(torch.stack([f.U.flatten(),f.V.flatten()],1))
            f.log_sigma.fill_(np.log(2*np.pi/max(h,w)/6));f.phi.zero_();f.log_amp.zero_();f.rho_raw.zero_()
            re,im=f._atoms_complex();psi=torch.complex(re,im).reshape(m,-1)
            mix=torch.linalg.solve(psi.T,target.reshape(d,-1).T).T
            f.w_real.copy_(mix.real);f.w_imag.copy_(mix.imag)
            error=float((f.build_filter()-target).norm()/target.norm())
            rows.append(dict(block=i,atoms=m,relative_frobenius_error=error,
                basis_condition_number=float(torch.linalg.cond(psi)),precision='float64',
                procedure='one localized Gaussian per distinct finite-grid bin; exact linear solve'))
    pd.DataFrame(rows).to_csv(ROOT/'results/tables/full_basis_capacity.csv',index=False)
    return rows
