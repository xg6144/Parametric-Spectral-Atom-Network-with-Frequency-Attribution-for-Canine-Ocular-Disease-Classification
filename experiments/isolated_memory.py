"""One model/input per fresh CUDA process; preserve separate latency sessions."""
import argparse
import json

import torch

from .common import ROOT,build_model,load_model,seed_all,sha256,write_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True);p.add_argument('--condition',required=True)
    p.add_argument('--batch',type=int,required=True);p.add_argument('--device',required=True)
    p.add_argument('--output',required=True);args=p.parse_args()
    seed_all(42);torch.set_num_threads(4)
    device=torch.device(args.device);torch.cuda.set_device(device)
    checkpoint=None
    if args.model=='psan':
        checkpoint=ROOT/json.loads((ROOT/'results/cache_equivalence.json').read_text())['checkpoint']
        model,_,_=load_model(checkpoint,device)
    else:model=build_model(args.model).to(device).eval()
    with torch.inference_mode():
        if args.condition=='cached':model.cache_filters(sha256(checkpoint))
        x=torch.randn(args.batch,3,224,224,device=device)
        for _ in range(3):model(x)
        torch.cuda.synchronize(device);torch.cuda.reset_peak_memory_stats(device)
        for _ in range(3):model(x)
        torch.cuda.synchronize(device)
    write_json(args.output,dict(model=args.model,condition=args.condition,batch=args.batch,
        device=args.device,peak_allocated_mb=torch.cuda.max_memory_allocated(device)/1024**2,
        peak_reserved_mb=torch.cuda.max_memory_reserved(device)/1024**2,
        units='MiB',scope='one model/input/cache and transient activations; fresh CUDA process',
        warmup=3,iterations=3,pytorch=torch.__version__,gpu=torch.cuda.get_device_name(device),
        latency_measured=False,checkpoint_sha256=sha256(checkpoint) if checkpoint else None))


if __name__=='__main__':main()
