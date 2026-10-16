from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .common import (ROOT,ManifestDataset,build_model,infer,load_manifest,load_model,
                     sha256,status,write_json)


def synchronize(device):
    if device.type=='cuda':torch.cuda.synchronize(device)


@torch.no_grad()
def measure(model,x,device,warmup=20,iterations=100):
    for _ in range(warmup):model(x)
    synchronize(device)
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    times=[]
    for _ in range(iterations):
        synchronize(device);tic=time.perf_counter();model(x);synchronize(device)
        times.append((time.perf_counter()-tic)*1000)
    return dict(latency_ms_median=float(np.median(times)),latency_ms_mean=float(np.mean(times)),
                latency_ms_sd=float(np.std(times,ddof=1)),throughput_images_sec=len(x)/np.mean(times)*1000,
                peak_allocated_mb=torch.cuda.max_memory_allocated(device)/1024**2 if device.type=='cuda' else None,
                peak_reserved_mb=torch.cuda.max_memory_reserved(device)/1024**2 if device.type=='cuda' else None,
                warmup=warmup,iterations=iterations)


def run(args):
    status('efficiency','IN PROGRESS')
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    checkpoint=args.checkpoint
    if not checkpoint:
        found=sorted((ROOT/'results/training').glob('canine_psan_gabor_ad0p2*/summary.json'))
        checkpoint=ROOT/json.loads(found[0].read_text())['checkpoint'] if found else next((ROOT/'train_results/sgkf5_fs42_psan_best_fold1_seed42').glob('*/best*.pth'))
    device=torch.device(args.device)
    if device.type=='cuda':torch.cuda.set_device(device)
    model,ckpt,_=load_model(checkpoint,device)
    checkpoint_hash=sha256(checkpoint)
    frame=load_manifest();frame=frame[frame.split=='test'].groupby('label',group_keys=False).sample(n=20,random_state=args.seed).sort_values('path')
    loader=DataLoader(ManifestDataset(frame),batch_size=64,num_workers=2)
    dynamic,y=infer(model,loader,device);model.cache_filters(checkpoint_hash);cached,yc=infer(model,loader,device)
    assert np.array_equal(y,yc)
    difference=float(np.max(np.abs(dynamic-cached)))
    if difference>1e-5:raise AssertionError('Cached logits exceed tolerance')
    agreement=float(np.mean(dynamic.argmax(-1)==cached.argmax(-1)))
    assert agreement==1
    equivalence=dict(checkpoint=str(Path(checkpoint).relative_to(ROOT)),checkpoint_sha256=checkpoint_hash,
        images=len(frame),max_logit_abs_difference=difference,argmax_agreement=agreement,
        accuracy_dynamic=float((dynamic.argmax(-1)==y).mean()),accuracy_cached=float((cached.argmax(-1)==y).mean()),
        scope='numerical equivalence on prespecified stratified sample; not independent legacy checkpoint accuracy',
        source_model_protocol=ckpt['config'].get('revision_protocol','validation-selected historical CV'))
    write_json(ROOT/'results/cache_equivalence.json',equivalence)
    rows=[]
    for name in ['psan','gfnet','afno']:
        m=model if name=='psan' else build_model(name).to(device).eval()
        for batch in [1,16,64]:
            x=torch.randn(batch,3,224,224,device=device)
            conditions=['dynamic','cached'] if name=='psan' else ['dynamic']
            for condition in conditions:
                if name=='psan':
                    m.clear_filter_cache()
                    if condition=='cached':m.cache_filters(checkpoint_hash)
                row=dict(model=name,condition=condition,device=str(device),batch=batch,
                    parameters=sum(p.numel() for p in m.parameters()),precision='float32',**measure(m,x,device))
                rows.append(row)
                print('Benchmark',row,flush=True)
        if name!='psan':del m
        else:model.to('cpu')
    cpu=model.to('cpu').eval();device_cpu=torch.device('cpu')
    for batch in [1,16]:
        x=torch.randn(batch,3,224,224)
        for condition in ['dynamic','cached']:
            cpu.clear_filter_cache()
            if condition=='cached':cpu.cache_filters(checkpoint_hash)
            rows.append(dict(model='psan',condition=condition,device='cpu',batch=batch,
                parameters=sum(p.numel() for p in cpu.parameters()),precision='float32',
                cpu_threads=4,**measure(cpu,x,device_cpu,warmup=10,iterations=30)))
    pd.DataFrame(rows).to_csv(ROOT/'results/tables/efficiency.csv',index=False)
    write_json(ROOT/'results/efficiency_protocol.json',dict(pytorch=torch.__version__,cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(torch.device(args.device)) if args.device.startswith('cuda') else None,
        cudnn=torch.backends.cudnn.version(),parameters_psan=sum(p.numel() for p in cpu.parameters()),
        input='synthetic float32 224x224 RGB',scope='model only; I/O and preprocessing excluded',
        synchronization='before and after every timed iteration',cpu_threads=4,
        fft_flops='not claimed: original fvcore number omits unsupported FFT/complex operations',
        cache_setup_cost_excluded=True,cache_equivalence=equivalence))
    (ROOT/'reports/10_computational_efficiency.md').write_text(f'''# Computational efficiency

Status: COMPLETED for dynamic/cached numerical equivalence and synchronized model-only GPU/CPU benchmarks.

Numerical equivalence: {equivalence}. The same selected trained checkpoint is used for both paths. All predictions agree; this is a function-preservation test, not a clinical validation claim. Raw data/split provenance limitations still apply to historical CV checkpoint scores.

Full results: `results/tables/efficiency.csv`. GPU batches1/16/64, fp32, 20 warmups/100 synchronized measurements; CPU batches1/16, four threads, 10 warmups/30 measurements. Peak GPU memory includes resident model/input, cache and activations after warmup; it is total allocated memory, not incremental cache memory. The cache-construction cost is excluded from steady-state latency and explicitly acknowledged. GFNet/AFNO use identical architecture/random input/runtime settings; runtime does not depend on their accuracy. Concurrent workload on other GPUs can affect host overhead; measurements are hardware/session-specific.

Cached filters are non-persistent and bound to a checkpoint SHA-256 in memory; checkpoint parameters remain the authoritative artifact. Training, loading, device/dtype conversion, parameter-version changes and explicit resolution adaptation invalidate caches. Gradient-enabled attribution always recomputes dynamic filters. Tests exercise cache mutation/state/resolution invalidation. Fixed resolution is required; interpolated positional embeddings change the model function and require new performance evaluation. Parameter count is measured directly. The original 0.744G profiler count is an incomplete supported-operation estimate, not a full FFT-inclusive FLOP count.
''')
    status('efficiency','COMPLETED',equivalence=equivalence,rows=len(rows))
