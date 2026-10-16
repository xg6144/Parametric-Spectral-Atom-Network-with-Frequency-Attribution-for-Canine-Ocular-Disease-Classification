"""Actual full-grid Gaussian/Gabor construction with determinant floor on/off."""
import json
import math

import pandas as pd
import torch

from models.psan import PSANConfig,PSANFilter
from .common import ROOT,sha256,write_json


def run():
    torch.set_num_threads(4);torch.manual_seed(42)
    path=next((ROOT/'train_results/gfnet_tiny').glob('*/best*.pth'))
    ckpt=torch.load(path,map_location='cpu',weights_only=False)
    key=next(k for k in ['eval_model','ema','model'] if ckpt.get(k) is not None)
    state=ckpt[key];rows=[]
    directory=ROOT/'results/capacity_construction';directory.mkdir(parents=True,exist_ok=True)
    for atom in ['gaussian','gabor']:
        for sigma in [.005,.01,.075]:
            saved=[]
            for i in range(12):
                target=torch.complex(state[f'blocks.{i}.mixer.filter_real'],state[f'blocks.{i}.mixer.filter_imag']).permute(2,0,1).to(torch.complex128)
                d,h,wh=target.shape;w=(wh-1)*2;m=h*wh
                filt=PSANFilter(PSANConfig(d,h,w,M=m,atom=atom,atom_dropout=0,anisotropic=True)).double().eval()
                with torch.no_grad():
                    filt.mu.copy_(torch.stack([filt.U.flatten(),filt.V.flatten()],1))
                    filt.log_sigma.fill_(math.log(sigma));filt.phi.zero_();filt.log_amp.zero_();filt.rho_raw.zero_();filt.theta.zero_()
                    re,im=filt._atoms_complex();psi=torch.complex(re,im).reshape(m,-1)
                    mixing=torch.linalg.solve(psi.T,target.reshape(d,-1).T).T
                    filt.w_real.copy_(mixing.real);filt.w_imag.copy_(mixing.imag)
                    residual=float((filt.build_filter()-target).norm()/target.norm())
                    if residual>1e-12:raise ValueError('Safeguarded constructive filter fit failed')
                    condition=float(torch.linalg.cond(psi))
                    saved.append(dict(block=i,config=vars(filt.cfg),state_dict=filt.state_dict()))
                determinant=float(math.exp(math.log(sigma))**4)
                rows.append(dict(atom=atom,sigma=sigma,block=i,atoms=m,determinant_before_floor=determinant,
                    floor_active=determinant<1e-8,effective_envelope_sigma=math.sqrt(max(determinant,1e-8)/(sigma*sigma)),
                    relative_frobenius_error=residual,basis_condition_number=condition))
            state_path=directory/f'{atom}_sigma{sigma}_fullgrid.pth';torch.save(saved,state_path)
            for row in rows[-12:]:row.update(fitted_states=str(state_path.relative_to(ROOT)),fitted_states_sha256=sha256(state_path))
    pd.DataFrame(rows).to_csv(ROOT/'results/tables/safeguarded_full_grid_capacity.csv',index=False)
    write_json(directory/'provenance.json',dict(target_checkpoint=str(path.relative_to(ROOT)),target_checkpoint_sha256=sha256(path),
        precision='CPU float64/complex128',target_state=key,source_code_sha256=sha256(__file__),
        protocol='M=F112, centers at actual FFT bins, rho0, common isotropic scale, theta0/phase0/amplitude1; exact complex mixing solve',
        image_or_test_label_use=False,scope='nominal filter tensor representability on this grid; not small-M or classification/generalization recovery'))
    print('Executed safeguarded full-grid constructions:',len(rows),'maximum residual',max(r['relative_frobenius_error'] for r in rows),flush=True)


if __name__=='__main__':run()
