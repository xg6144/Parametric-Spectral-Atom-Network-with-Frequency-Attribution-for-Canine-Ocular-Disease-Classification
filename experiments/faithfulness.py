"""Actual atom/frequency interventions on a fixed block, target and budget."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

from models.psan import _forward_to_block,_forward_from_block,spectral_cam
from .common import ROOT,CLASSES,ManifestDataset,load_manifest,load_model,seed_all,sha256,status,write_json
from .analysis import savefig


def suffix(model,h_in,output,index):
    block=model.blocks[index]
    mixer=h_in+output
    final=mixer+block.mlp(block.norm2(mixer))
    return _forward_from_block(model,final,index+1)


def attributions(model,x,index,target):
    with torch.no_grad():
        h=_forward_to_block(model,x,index)
        c=model.blocks[index].filter.per_atom_contribution(model.blocks[index].norm1(h))
    output=c.sum(1).detach().requires_grad_(True)
    logits=suffix(model,h,output,index)
    gradient=torch.autograd.grad(logits.gather(1,target[:,None]).sum(),output)[0]
    attr=(gradient[:,None]*c).sum((2,3,4))
    return attr.detach(),c.detach(),h.detach(),logits.detach(),gradient.detach()


def safe_spearman(a,b):
    if np.ptp(a)==0 or np.ptp(b)==0:return np.nan
    return float(spearmanr(a,b).statistic)


def mean_ci(values,groups,seed=42,iterations=1000):
    values=np.asarray(values,dtype=float);groups=np.asarray(groups)
    keep=np.isfinite(values);values,groups=values[keep],groups[keep]
    if len(values)==0:return dict(mean=None,ci_low=None,ci_high=None,n=0)
    _,codes=np.unique(groups,return_inverse=True);n=int(codes.max()+1)
    sums=np.bincount(codes,weights=values);counts=np.bincount(codes)
    rng=np.random.default_rng(seed);out=[]
    for _ in range(iterations):
        sample=rng.integers(0,n,n);out.append(sums[sample].sum()/counts[sample].sum())
    return dict(mean=float(values.mean()),ci_low=float(np.quantile(out,.025)),ci_high=float(np.quantile(out,.975)),n=len(values))


def frequency_groups(h,w):
    groups=[];wh=w//2+1
    for v in range(wh):
        boundary=v==0 or (w%2==0 and v==w//2)
        seen=set()
        for u in range(h):
            if u in seen:continue
            cells=[(u,v),((-u)%h,v)] if boundary and (-u)%h!=u else [(u,v)]
            seen.update(a for a,_ in cells);groups.append(cells)
    return groups


def spectrum_interventions(model,h,c,gradient,index,target,rng):
    out=c.sum(1);b,height,width,d=out.shape
    block=model.blocks[index]
    spectrum=torch.fft.rfft2(out,dim=(1,2),norm='ortho')
    input_spec=torch.fft.rfft2(block.norm1(h),dim=(1,2),norm='ortho')
    alpha=gradient.mean((1,2))[:,None,None,:]
    maps={'input':((input_spec*alpha).sum(-1).abs()**2).cpu().numpy(),
          'filtered':((spectrum*alpha).sum(-1).abs()**2).cpu().numpy()}
    grad_spec=torch.fft.rfft2(gradient,dim=(1,2),norm='ortho')
    multiplicity=torch.full((width//2+1,),2.,device=out.device)
    multiplicity[0]=1
    if width%2==0:multiplicity[-1]=1
    signed=(grad_spec.conj()*spectrum).real.sum(-1)*multiplicity[None,None,:]
    torch.testing.assert_close(signed.sum((1,2)),(gradient*out).sum((1,2,3)),atol=2e-5,rtol=1e-4)
    maps['signed_gradient_response']=signed.cpu().numpy()
    groups=frequency_groups(height,width);budget=[1,10,25,50]
    scores={name:np.stack([value[:,[u for u,v in cells],[v for u,v in cells]].sum(1) for cells in groups],axis=1)
            for name,value in maps.items()}
    base=suffix(model,h,out,index).gather(1,target[:,None])[:,0].cpu().numpy()
    rows=[]
    for name in ['input','filtered','signed_gradient_response','random']:
        order=np.argsort(-scores[name],axis=1) if name!='random' else np.stack([rng.permutation(len(groups)) for _ in range(b)])
        for percent in budget:
            k=max(1,round(percent/100*len(groups)));mask=torch.ones(b,height,width//2+1,1,device=out.device)
            for i in range(b):
                for g in order[i,:k]:
                    for u,v in groups[g]:mask[i,u,v]=0
            removed=torch.fft.irfft2(spectrum*mask,s=(height,width),dim=(1,2),norm='ortho')
            score=suffix(model,h,removed,index).gather(1,target[:,None])[:,0].cpu().numpy()
            for i in range(b):rows.append(dict(batch_row=i,method=name,budget_percent=percent,
                 removed_frequency_groups=k,total_frequency_groups=len(groups),logit_drop=float(base[i]-score[i])))
    return rows


def explanation_examples(model,dataset,frame,directory,device):
    candidates=[]
    for cls in ['Nuclear_sclerosis','Eyelid_tumor','Non_ulcerative_keratitis','Pigmentary_keratitis','Epiphora']:
        idx=CLASSES.index(cls)
        found={True:None,False:None}
        for i in np.flatnonzero(frame.label.to_numpy()==idx):
            x,y=dataset[int(i)];x=x[None].to(device)
            with torch.no_grad():p=model(x).softmax(-1)[0].cpu().numpy()
            correct=int(p.argmax())==y
            if found[correct] is None:found[correct]=(int(i),p)
            if all(v is not None for v in found.values()):break
        candidates.extend((cls,k,v) for k,v in found.items() if v is not None)
    if not candidates:return
    fig,axes=plt.subplots(len(candidates),5,figsize=(16,2.6*len(candidates)),squeeze=False)
    records=[];panels=[]
    for row,(cls,correct,(i,p)) in enumerate(candidates):
        x,y=dataset[i];raw=(x.permute(1,2,0).numpy()*np.array([.229,.224,.225])+np.array([.485,.456,.406])).clip(0,1)
        x=x[None].to(device);target=int(p.argmax());index=len(model.blocks)-1
        with torch.no_grad():
            h=_forward_to_block(model,x,index);out=model.blocks[index].filter(model.blocks[index].norm1(h));mixed=h+out
            features=(mixed+model.blocks[index].mlp(model.blocks[index].norm2(mixed))).detach()
        features.requires_grad_(True);logits=_forward_from_block(model,features,index+1)
        grad=torch.autograd.grad(logits[0,target],features)[0];weights=grad.mean((1,2),keepdim=True)
        cam=torch.relu((features*weights).sum(-1)).detach()
        cam=torch.nn.functional.interpolate(cam[:,None],size=(224,224),mode='bilinear',align_corners=False)[0,0].cpu().numpy()
        cam=(cam-cam.min())/(cam.max()-cam.min()+1e-12)
        old=spectral_cam(model,x,target,response='input')[0].detach().cpu().numpy()
        filtered=spectral_cam(model,x,target,response='filtered')[0].detach().cpu().numpy()
        attr,_,_,_,_=attributions(model,x,index,torch.tensor([target],device=device));values=attr[0].cpu().numpy()
        axes[row,0].imshow(raw);axes[row,0].set_title(f'{cls.replace("_"," ")}\nPred: {CLASSES[target].replace("_"," ")} ({p[target]:.3f})',fontsize=9)
        axes[row,1].imshow(raw);axes[row,1].imshow(cam,cmap='inferno',alpha=.45,vmin=0,vmax=1);axes[row,1].set_title('Grad-CAM: final block output',fontsize=9)
        for col,value,name in [(2,old,'Legacy input spectrum'),(3,filtered,'Actual filtered spectrum')]:
            im=axes[row,col].imshow(np.log1p(np.fft.fftshift(value,axes=0)),cmap='viridis',aspect='auto');axes[row,col].set_title(name,fontsize=9);fig.colorbar(im,ax=axes[row,col],fraction=.05,label='log(1+score)')
            axes[row,col].set(xlabel='Nonnegative width-frequency bin',ylabel='Shifted height-frequency bin')
        axes[row,4].bar(range(len(values)),values,color=np.where(values>=0,'#4477aa','#cc6677'));axes[row,4].set(xlabel='Atom index',ylabel='Signed local attribution')
        for col in [0,1]:axes[row,col].axis('off')
        records.append(dict(path=frame.iloc[i].path,true_class=cls,predicted_class=CLASSES[target],
                            correct=correct,probabilities=dict(zip(CLASSES,p.tolist())),target='predicted class',
                            gradcam_layer='output of final PSAN block before final LayerNorm/GAP'))
        panels.append(dict(cls=cls,path=frame.iloc[i].path,correct=correct,p=p,target=target,
                           raw=raw,cam=cam,filtered=filtered,values=values))
    fig.tight_layout();savefig(fig,ROOT/'results/figures'/(directory.name+'_real_examples'))
    write_json(directory/'examples.json',records)
    # Publication panels: one correct/incorrect pair per class, sufficient
    # physical label size at full IEEE page width; full atlas remains archived.
    for cls in sorted({p['cls'] for p in panels}):
        chosen=[p for p in panels if p['cls']==cls]
        fig,axes=plt.subplots(len(chosen),4,figsize=(7.2,1.95*len(chosen)),squeeze=False)
        for row,p in enumerate(chosen):
            axes[row,0].imshow(p['raw']);axes[row,0].axis('off')
            predicted=CLASSES[p['target']].replace('_',' ')
            axes[row,0].set_title(('Correct' if p['correct'] else 'Incorrect')+'\n'+predicted+f' p={p["p"][p["target"]]:.3f}',fontsize=8)
            axes[row,1].imshow(p['raw']);axes[row,1].imshow(p['cam'],cmap='inferno',alpha=.45,vmin=0,vmax=1)
            axes[row,1].axis('off');axes[row,1].set_title('Spatial Grad-CAM',fontsize=8)
            energy=np.log1p(np.fft.fftshift(p['filtered'],axes=0))
            normalized_energy=energy/max(float(energy.max()),1e-30)
            im=axes[row,2].imshow(normalized_energy,cmap='viridis',aspect='auto',vmin=0,vmax=1)
            axes[row,2].set_title('Filtered Spectral-CAM',fontsize=8)
            axes[row,2].set_xlabel('Width-frequency bin',fontsize=7);axes[row,2].set_ylabel('Shifted height bin',fontsize=7)
            axes[row,2].tick_params(labelsize=7);bar=fig.colorbar(im,ax=axes[row,2],fraction=.06,ticks=[0,.5,1]);bar.ax.tick_params(labelsize=7)
            axes[row,3].bar(range(16),p['values'],color=np.where(p['values']>=0,'#4477aa','#cc6677'))
            axes[row,3].set_title('Signed atom scores',fontsize=8);axes[row,3].set_xlabel('Atom index',fontsize=7);axes[row,3].tick_params(labelsize=7)
        fig.tight_layout();savefig(fig,ROOT/'results/figures'/(directory.name+'_examples_'+cls))


def deletion_figure(curve,directory):
    fig,axes=plt.subplots(1,3,figsize=(7.2,2.8),layout='constrained')
    colors={'lowest_absolute':'#4477aa','lowest_signed':'#ccbb44','random':'#228833','top_signed':'#aa3377'}
    styles={'lowest_absolute':':','lowest_signed':'--','random':'-.','top_signed':'-'}
    for ax,index in zip(axes,[0,5,11]):
        for method,part in curve[curve.block==index].groupby('method'):
            mean=part.groupby('k').logit_drop.mean()
            ax.plot(mean.index,mean.values,label=method.replace('_',' '),color=colors[method],ls=styles[method])
        ax.set_title('Block '+str(index),fontsize=10)
        ax.set_xlabel('Atoms removed',fontsize=9);ax.tick_params(labelsize=8)
    axes[0].set_ylabel('Fixed target logit drop',fontsize=9)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,1.12),ncol=4,fontsize=8)
    savefig(fig,ROOT/'results/figures'/(directory.name+'_deletion'))


def run(args):
    status('attribution','IN PROGRESS');torch.set_num_threads(4);seed_all(args.seed)
    device=torch.device(args.device)
    checkpoint=args.checkpoint
    if not checkpoint:
        found=sorted((ROOT/'results/training').glob('canine_psan_gabor_ad0p2*/summary.json'))
        checkpoint=ROOT/json.loads(found[0].read_text())['checkpoint'] if found else next((ROOT/'train_results/sgkf5_fs42_psan_best_fold1_seed42').glob('*/best*.pth'))
    checkpoint=Path(checkpoint).resolve()
    model,ckpt,_=load_model(checkpoint,device);digest=sha256(checkpoint)
    directory=ROOT/'results/attribution'/digest[:12];directory.mkdir(parents=True,exist_ok=True)
    frame=load_manifest();frame=frame[frame.split=='test'].groupby('label',group_keys=False).sample(n=args.samples_per_class,random_state=args.seed).sort_values('path').reset_index(drop=True)
    frame.to_csv(directory/'sample_manifest.csv',index=False);dataset=ManifestDataset(frame)
    loader=DataLoader(dataset,batch_size=8,num_workers=2,shuffle=False)
    removal=[];curves=[];sanity=[];spectral=[];branch_checks=[];offset=0;rng=np.random.default_rng(args.seed)
    randomized=[]
    for level in ['head','tail','full']:
        other=copy.deepcopy(model)
        modules=[other.head] if level=='head' else [other.head,other.blocks[-1].mlp] if level=='tail' else list(other.modules())
        seen=set()
        for module in modules:
            for sub in module.modules():
                if id(sub) in seen:continue
                seen.add(id(sub))
                if isinstance(sub,(torch.nn.Linear,torch.nn.LayerNorm)):sub.reset_parameters()
        if level=='full':
            with torch.no_grad():
                other.patch_embed.proj.reset_parameters()
                torch.nn.init.trunc_normal_(other.pos_embed,std=.02)
                for block in other.blocks:
                    block.filter.w_real.normal_(0,.02);block.filter.w_imag.normal_(0,.02)
        randomized.append((level,other.eval()))
    for x,y in loader:
        x=x.to(device)
        with torch.no_grad():base_logits=model(x);target=base_logits.argmax(-1)
        batch=len(x)
        for index in [0,5,11]:
            attr,c,h,logits,gradient=attributions(model,x,index,target)
            assert torch.allclose(logits,base_logits,atol=2e-5,rtol=1e-4)
            values=attr.cpu().numpy();m=c.shape[1];base=logits.gather(1,target[:,None])[:,0].cpu().numpy()
            with torch.no_grad():
                deltas=[]
                for atom in range(m):
                    mask=torch.ones(m,device=device);mask[atom]=0
                    native_output=model.blocks[index].filter(model.blocks[index].norm1(h),atom_mask=mask)
                    difference=float((native_output-(c.sum(1)-c[:,atom])).abs().max())
                    if difference>1e-5:raise AssertionError('Native atom removal and feature decomposition disagree')
                    branch_checks.append(dict(block=index,atom=atom,batch_offset=offset,max_abs_difference=difference))
                    perturbed=suffix(model,h,native_output,index)
                    deltas.append(base-perturbed.gather(1,target[:,None])[:,0].cpu().numpy())
                deltas=np.stack(deltas,axis=1)
                orders={'top_signed':np.argsort(-values,axis=1),'lowest_signed':np.argsort(values,axis=1),
                        'lowest_absolute':np.argsort(np.abs(values),axis=1),
                        'random':np.stack([rng.permutation(m) for _ in range(batch)])}
                for method,order in orders.items():
                    remaining=c.sum(1).clone()
                    for k in range(1,m+1):
                        chosen=torch.as_tensor(order[:,k-1],device=device)
                        remaining-=c[torch.arange(batch,device=device),chosen]
                        if k==m:remaining.zero_()
                        scores=suffix(model,h,remaining,index)
                        drop=base-scores.gather(1,target[:,None])[:,0].cpu().numpy()
                        confidence=scores.softmax(-1).gather(1,target[:,None])[:,0].cpu().numpy()
                        for j in range(batch):curves.append(dict(path=frame.iloc[offset+j].path,group_id=frame.iloc[offset+j].group_id,
                            block=index,method=method,k=k,target=int(target[j]),logit_drop=float(drop[j]),confidence=float(confidence[j])))
                for j in range(batch):
                    removal.append(dict(path=frame.iloc[offset+j].path,group_id=frame.iloc[offset+j].group_id,block=index,
                        true_label=int(y[j]),target=int(target[j]),spearman=safe_spearman(values[j],deltas[j]),
                        sign_agreement=float((np.sign(values[j])==np.sign(deltas[j])).mean()),
                        positive_atoms=int((values[j]>0).sum()),negative_atoms=int((values[j]<0).sum()),
                        attribution=values[j].tolist(),actual_logit_drops=deltas[j].tolist()))
                if index==11:
                    local=spectrum_interventions(model,h,c,gradient,index,target,rng)
                    for r in local:
                        j=r.pop('batch_row');spectral.append(dict(path=frame.iloc[offset+j].path,group_id=frame.iloc[offset+j].group_id,**r))
            if index==11:
                for level,other in randomized:
                    random_attr,*_=attributions(other,x,index,target)
                    random_values=random_attr.cpu().numpy()
                    for j in range(batch):sanity.append(dict(path=frame.iloc[offset+j].path,group_id=frame.iloc[offset+j].group_id,
                       intervention=level+'_parameter_randomization',correlation=safe_spearman(values[j],random_values[j])))
                noise=torch.randn_like(x)*.01/torch.tensor([.229,.224,.225],device=device)[None,:,None,None]
                noise_attr,*_=attributions(model,x+noise,index,target)
                second=base_logits.argsort(-1,descending=True)[:,1];class_attr,*_=attributions(model,x,index,second)
                for j in range(batch):
                    for name,reference in [('rgb_noise_sigma0p01',noise_attr[j].cpu().numpy()),('alternative_runner_up_class',class_attr[j].cpu().numpy()),('random_attribution',rng.normal(size=m))]:
                        sanity.append(dict(path=frame.iloc[offset+j].path,group_id=frame.iloc[offset+j].group_id,intervention=name,correlation=safe_spearman(values[j],reference)))
        offset+=batch
        print('Attribution audited',offset,'/',len(frame),flush=True)
    pd.DataFrame(removal).drop(columns=['attribution','actual_logit_drops']).to_csv(directory/'atom_removal.csv',index=False)
    pd.DataFrame(branch_checks).to_csv(directory/'native_mask_equivalence.csv',index=False)
    write_json(directory/'atom_values.json',removal);curve=pd.DataFrame(curves);curve.to_csv(directory/'deletion_curves.csv',index=False)
    pd.DataFrame(sanity).to_csv(directory/'sanity.csv',index=False);spec=pd.DataFrame(spectral);spec.to_csv(directory/'spectral_deletion.csv',index=False)
    stats=[]
    for index,part in pd.DataFrame(removal).groupby('block'):
        for metric in ['spearman','sign_agreement']:stats.append(dict(block=index,metric=metric,**mean_ci(part[metric],part.group_id,args.seed,args.bootstrap)))
    for (index,method),part in curve.groupby(['block','method']):
        aopc=part.groupby(['path','group_id']).logit_drop.mean().reset_index()
        stats.append(dict(block=index,metric='AOPC_logit_'+method,**mean_ci(aopc.logit_drop,aopc.group_id,args.seed,args.bootstrap)))
    for intervention,part in pd.DataFrame(sanity).groupby('intervention'):
        stats.append(dict(block=11,metric=intervention,**mean_ci(part.correlation,part.group_id,args.seed,args.bootstrap)))
    for (method,budget),part in spec.groupby(['method','budget_percent']):
        stats.append(dict(block=11,metric=f'spectral_{method}_{budget}percent_logit_drop',**mean_ci(part.logit_drop,part.group_id,args.seed,args.bootstrap)))
    pd.DataFrame(stats).to_csv(directory/'summary.csv',index=False)
    deletion_figure(curve,directory)
    explanation_examples(model,dataset,frame,directory,device)
    provenance=dict(checkpoint=str(Path(checkpoint).relative_to(ROOT)),checkpoint_sha256=digest,
        active_split_sha256=ckpt['config'].get('split_sha256'),source_code_sha256=sha256(__file__),
        images=len(frame),samples_per_class=args.samples_per_class,blocks=[0,5,11],target='original predicted class fixed throughout',
        atom_removal='one block only; exactly zero removed atoms; no survivor rescaling; all other blocks unchanged',
        aopc='mean logit drop over k=1..16 per image, then mean across prespecified balanced sample',
        spectral_deletion='actual real filter response rFFT; remove conjugate-consistent boundary groups with identical budgets',
        signed_frequency_comparator='weighted real-FFT gradient-response inner product; sums to local gradient dot output, not final logit',
        full_randomization='all dense, normalization, patch projection, positional embedding and mixing weights; atom geometry kept for index comparability',
        noise_sigma_rgb=.01,bootstrap=args.bootstrap,cohort=ckpt['config'].get('revision_protocol','historical validation-selected CV'),
        scope='balanced diagnostic subset; not lesion localization ground-truth or clinical faithfulness')
    write_json(directory/'provenance.json',provenance)
    if str(checkpoint).startswith(str(ROOT/'results/training')):
        from .attribution_diagnostics import run as additional_diagnostics
        additional_diagnostics(directory)
    (ROOT/'reports/05_attribution_faithfulness.md').write_text(f'''# Frequency attribution faithfulness

Status: COMPLETED for the executed checkpoint/subset: `{provenance}`.

Artifacts: `{directory.relative_to(ROOT)}` (sample manifest, exact atom values, real single-atom removal deltas, deletion curves, sanity checks, spectral interventions and group-bootstrap summary). Layer indices are zero-based. The target is the original predicted class and stays fixed even when the perturbed model's argmax changes. Removing an atom means setting its mixing contribution to zero in one specified block; there is no renormalization and no change to other blocks. The prefix is held fixed; the complete nonlinear suffix is rerun for every intervention. Equal removal budgets apply to top-signed, lowest-signed, lowest-absolute and random rankings. Lowest-signed removes negative atoms when they exist; counts are explicitly recorded. AOPC averages all k1–16 logit changes, not a cherry-picked budget. Positive local scores need not predict finite interventions exactly through a nonlinear suffix.

Spearman is computed per image across actual 16 atom logit changes; constant vectors are undefined and excluded with effective n reported. Sign agreement is the mean atom-wise signed agreement. Bootstrap resamples visual groups on the fixed balanced subset. It does not quantify training-seed variability or unobserved dog dependence. Parameter randomization preserves atom identity/geometry and separately resets head, tail MLP/head, or all dense/normalization/mixing weights. Noise and runner-up-class correlations describe sensitivity/conditional differences; repeated stability and label-randomization TRAINING are not claimed here.

The legacy Spectral-CAM uses the pre-filter input spectrum. The revised filtered variant uses rFFT of the actual real filter output, respecting implicit boundary projection. Both use spatial-mean channel gradients and squared weighted complex magnitude: nonnegative summaries, not signed or complete logit explanations. Frequency deletion tests remove conjugate-consistent boundary groups with fixed budgets1/10/25/50 percent from the actual filter response; comparisons include random frequency groups. These tests assess finite target-logit behavior, not human clinical plausibility. Correct and incorrect examples are real dataset crops with all class probabilities, target/layer labels and vector figures. No clinical causal interpretation is asserted without expert/lesion-mask validation.

Mathematically, sum_m c_m=Z in the linear filter branch. Sum_m <grad_f(Z),c_m>=<grad_f(Z),Z> is a local directional derivative; it generally differs from f_c(x). Exact branch decomposition must not be called exact final-logit completeness. References: [Adebayo et al., NeurIPS2018](https://papers.neurips.cc/paper_files/paper/2018/hash/294a8ed24b1ad22ec2e7efea049b8737-Abstract.html).
''')
    status('attribution','COMPLETED',provenance=provenance,summary=str((directory/'summary.csv').relative_to(ROOT)))
