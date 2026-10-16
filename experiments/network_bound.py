"""Conservative fixed-parameter evaluation Lipschitz estimates, not certificates."""
import math

import pandas as pd
import torch

from .common import ROOT,load_model,sha256,write_json


def run():
    torch.set_num_threads(4);rows=[];blocks=[]
    lg=1+1/math.sqrt(2*math.pi*math.e)
    paths=[ROOT/'results/training'/f'canine_psan_gabor_ad0p2_M16_morlet_s{s}/best.pth' for s in [42,43,44,45,46]]
    paths+=[ROOT/'results/tuning/final/canine_psan_c2_s42_final/best.pth']
    def frobenius(layer):return float(torch.linalg.vector_norm(layer.weight.double()))
    def norm_bound(layer):return float(layer.weight.double().abs().max())/math.sqrt(layer.eps)
    for path in paths:
        model,_,_=load_model(path,'cpu');model.double().eval()
        if not isinstance(model.pre_logits,torch.nn.Identity):raise ValueError('Bound assumes identity representation head')
        projection=model.patch_embed.proj
        if projection.kernel_size!=projection.stride or projection.padding!=(0,0):raise ValueError('Patch bound assumes disjoint unpadded patches')
        log_bound=math.log10(frobenius(projection)*norm_bound(model.norm)*frobenius(model.head)/
                              math.sqrt(model.patch_embed.grid_h*model.patch_embed.grid_w))
        for i,block in enumerate(model.blocks):
            if block.mlp.act.approximate!='none':raise ValueError('GELU bound uses exact Gaussian-CDF activation')
            with torch.no_grad():
                k=block.filter.build_filter();gain=float(k.abs().max())
                w=torch.complex(block.filter.w_real,block.filter.w_imag)
                coefficient=float((w.abs()*block.filter.log_amp.exp()[None,:]).sum(1).max())
            first=1+gain*norm_bound(block.norm1)
            second=1+lg*frobenius(block.mlp.fc1)*frobenius(block.mlp.fc2)*norm_bound(block.norm2)
            log_bound+=math.log10(first)+math.log10(second)
            blocks.append(dict(checkpoint=str(path.relative_to(ROOT)),block=i,nominal_filter_gain=gain,
                coefficient_envelope_bound=coefficient,filter_residual_bound=first,mlp_residual_bound=second))
        rows.append(dict(checkpoint=str(path.relative_to(ROOT)),checkpoint_sha256=sha256(path),
                        log10_frobenius_based_network_bound=log_bound))
    pd.DataFrame(rows).to_csv(ROOT/'results/tables/network_stability_estimates.csv',index=False)
    pd.DataFrame(blocks).to_csv(ROOT/'results/tables/filter_gain_estimates.csv',index=False)
    write_json(ROOT/'results/network_bound_protocol.json',dict(source_code_sha256=sha256(__file__),
        domain='fixed224 standardized model inputs; evaluation mode; fixed finite weights and positive LayerNorm epsilon',
        weight_bound='Frobenius norm upper bounds each real linear spectral norm',
        layernorm_bound='max(abs(gamma))/sqrt(epsilon), proved from its centered normalization Jacobian',
        exact_gelu_bound=lg,
        interpretation='FP64 numerical estimates of a conservative analytic product; not rounding-certified adversarial or clinical robustness',
        training_guarantee=False))
    print('Fixed-parameter network bound estimates',rows,flush=True)


if __name__=='__main__':run()
