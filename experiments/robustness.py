from __future__ import annotations

import hashlib
import io
import json

import numpy as np
import pandas as pd
import torch
from PIL import Image,ImageEnhance,ImageFilter
from torch.utils.data import Dataset,DataLoader

from evaluation_metrics import compute_metric_bundle
from .common import ROOT,ManifestDataset,load_manifest,load_model,status,transform,write_json


SEVERITIES={'gaussian_noise':[.01,.03,.05],'gaussian_blur':[.5,1.,2.],
             'jpeg_quality':[80,50,20],'brightness_factor':[.7,.5,.3],
             'contrast_factor':[.7,.5,.3],'saturation_factor':[.7,.4,.1]}


def corrupt(image,name,level,seed):
    value=SEVERITIES[name][level-1]
    if name=='gaussian_noise':
        rng=np.random.default_rng(seed);a=np.asarray(image,dtype=np.float32)/255
        a=np.clip(a+rng.normal(0,value,a.shape),0,1)
        return Image.fromarray(np.round(a*255).astype(np.uint8))
    if name=='gaussian_blur':return image.filter(ImageFilter.GaussianBlur(value))
    if name=='jpeg_quality':
        buffer=io.BytesIO();image.save(buffer,format='JPEG',quality=value,subsampling=2)
        buffer.seek(0)
        with Image.open(buffer) as result:return result.convert('RGB').copy()
    if name=='brightness_factor':return ImageEnhance.Brightness(image).enhance(value)
    if name=='contrast_factor':return ImageEnhance.Contrast(image).enhance(value)
    if name=='saturation_factor':return ImageEnhance.Color(image).enhance(value)
    raise ValueError(name)


class CorruptionDataset(Dataset):
    def __init__(self,frame,name,level,seed):
        self.base=ManifestDataset(frame);self.frame=frame.reset_index(drop=True)
        self.name,self.level,self.seed=name,level,seed;self.transform=transform(False,224)
    def __len__(self):return len(self.frame)
    def __getitem__(self,i):
        row=self.frame.iloc[i]
        if self.base.cached is not None and self.base.cache_indices[i]>=0:
            image=Image.fromarray(self.base.cached[self.base.cache_indices[i]])
        else:
            with Image.open(ROOT/'data_raw'/row.path) as img:image=img.convert('RGB').resize((224,224),Image.Resampling.BILINEAR)
        if self.name!='clean':
            seed=int(hashlib.sha256((row.path+'|'+self.name+'|'+str(self.level)+'|'+str(self.seed)).encode()).hexdigest()[:16],16)
            image=corrupt(image,self.name,self.level,seed)
        return self.transform(image),int(row.label)


def run(args):
    status('robustness','IN PROGRESS');torch.set_num_threads(4)
    device=torch.device(args.device);models={};provenance={}
    for name in args.models:
        found=[json.loads(p.read_text()) for p in (ROOT/'results/training').glob('*/summary.json')
               if json.loads(p.read_text())['dataset']=='canine' and json.loads(p.read_text())['model']==name
               and json.loads(p.read_text())['seed']==args.seed
               and (name!='psan' or (json.loads(p.read_text())['atom']=='gabor' and json.loads(p.read_text())['atom_dropout']==.2))]
        if len(found)!=1:raise ValueError(f'Need one completed main {name}/seed{args.seed} run; found {len(found)}')
        s=found[0];models[name],_,_=load_model(ROOT/s['checkpoint'],device);provenance[name]=s['checkpoint_sha256']
    frame=load_manifest();frame=frame[frame.split=='test'].reset_index(drop=True)
    classes=sorted(frame['class'].unique());rows=[];directory=ROOT/'results/robustness';directory.mkdir(exist_ok=True)
    frame[['path','label','group_id']].to_csv(directory/'sample_manifest.csv',index=False)
    conditions=[('clean',0)]+[(name,i) for name in SEVERITIES for i in [1,2,3]]
    clean={}
    for name,level in conditions:
        probabilities={m:[] for m in models};labels=[]
        loader=DataLoader(CorruptionDataset(frame,name,level,args.seed),batch_size=128,num_workers=6,shuffle=False)
        with torch.no_grad():
            for x,y in loader:
                x=x.to(device);labels.append(y.numpy())
                for m,model in models.items():probabilities[m].append(model(x).softmax(-1).cpu().numpy())
        y=np.concatenate(labels);assert np.array_equal(y,frame.label.to_numpy())
        for m,values in probabilities.items():
            p=np.concatenate(values);bundle=compute_metric_bundle(y,p,classes)
            np.savez_compressed(directory/f'{m}_{name}_s{level}.npz',probabilities=p,labels=y)
            metric=bundle['metrics']
            if name=='clean':clean[m]=metric
            row=dict(model=m,seed=args.seed,corruption=name,severity=level,
                parameter=0 if name=='clean' else SEVERITIES[name][level-1],images=len(y),
                accuracy=metric['accuracy'],macro_f1=metric['macro_f1'],macro_auc_ovr=metric['macro_auc_ovr'],
                macro_f1_absolute_drop=clean[m]['macro_f1']-metric['macro_f1'],
                macro_f1_relative_drop_percent=100*(clean[m]['macro_f1']-metric['macro_f1'])/clean[m]['macro_f1'])
            rows.append(row);print('Corruption',row,flush=True)
        pd.DataFrame(rows).to_csv(ROOT/'results/tables/robustness.csv',index=False)
    from .analysis import savefig
    import matplotlib.pyplot as plt
    table=pd.DataFrame(rows);fig,axes=plt.subplots(2,3,figsize=(11,6))
    for ax,name in zip(axes.flat,SEVERITIES):
        for model,part in table[table.corruption==name].groupby('model'):
            ax.plot(part.severity,part.macro_f1,'o-',label=model)
        ax.set(title=name.replace('_',' '),xlabel='Severity',ylabel='Macro F1',xticks=[1,2,3]);ax.legend(fontsize=8)
    fig.tight_layout();savefig(fig,ROOT/'results/figures/corruption_robustness')
    write_json(directory/'protocol.json',dict(seed=args.seed,images=len(frame),severities=SEVERITIES,
        checkpoint_hashes=provenance,noise_seed='SHA256(path|corruption|level|seed) first64bits',
        preprocessing='Resize to224 before corruption; ImageNet normalization afterward',
        jpeg_subsampling=2,noise_quantization='round clipped RGB[0,1] to uint8',
        saturation='desaturation via PIL ImageEnhance.Color',confidence_intervals='not computed for corruption table',
        model_uncertainty='one prespecified seed42; not across-seed robustness',
        spatial_corruption_not_real_capture_shift=True))
    (ROOT/'reports/07_robustness.md').write_text(f'''# Image quality corruption robustness

Status: COMPLETED for PSAN/GFNet/AFNO seed{args.seed}, all {len(frame):,} final-test images, six corruptions × three severities plus clean.

`robustness.csv` contains accuracy, Macro F1, macro OvR AUC, clean-minus-corrupt F1 and relative drop100*(clean−corrupt)/clean. Negative drops (improvements) are retained. Severity definitions={SEVERITIES}. JPEG uses chroma subsampling2. Gaussian noise is applied in RGB[0,1], clipped and rounded to uint8, with the same deterministic per-path draw for every model. Corruptions occur after deterministic224 resizing and before normalization. Saved NPZ probabilities/labels plus the immutable sample manifest regenerate all metrics and vector/300-dpi figures.

Every model receives exactly the same corrupted images. This is a synthetic acquisition-quality stress test on one prespecified seed, not a new real device/hospital/patient domain or proof of clinical robustness. No across-seed corruption SD or corruption bootstrap CI is claimed. All severities, including unfavorable results, are reported. Frequency-model hypotheses about low/high-frequency sensitivity are interpreted using the actual observed curves, not assumed advantages.
''')
    status('robustness','COMPLETED',models=list(models),images=len(frame),conditions=19)
