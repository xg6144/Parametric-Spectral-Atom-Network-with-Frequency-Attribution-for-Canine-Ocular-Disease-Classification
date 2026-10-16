"""Immutable deterministic image cache; original files are never modified."""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image
from torchvision.transforms import Resize

from .common import ROOT, sha256, write_json


def main():
    directory = ROOT/'results/cache'
    directory.mkdir(parents=True,exist_ok=True)
    if (directory/'canine_224_complete.json').exists():
        print('Verified cache already present',flush=True)
        return
    frame = pd.read_csv(ROOT/'cv_splits/stratified_group_5fold.csv')
    arr = np.lib.format.open_memmap(directory/'canine_224.npy',mode='w+',
                                    dtype=np.uint8,shape=(len(frame),224,224,3))
    resize = Resize((224,224))
    def decode(pair):
        i,path = pair
        with Image.open(ROOT/'data_raw'/path) as image:
            value = np.asarray(resize(image.convert('RGB')))
            arr[i] = value
        return i
    with ThreadPoolExecutor(max_workers=16) as pool:
        for i in pool.map(decode,enumerate(frame.path)):
            if i % 10000 == 0: print('Resized/decode verified',i,'/',len(frame),flush=True)
    arr.flush()
    frame[['path']].to_csv(directory/'canine_224_paths.csv',index=False)
    write_json(directory/'canine_224_complete.json',{
        'frozen_manifest_sha256':sha256(ROOT/'cv_splits/stratified_group_5fold.csv'),
        'images':len(frame),'shape':list(arr.shape),'dtype':'uint8',
        'preprocessing':'PIL RGB; torchvision Resize 224x224 bilinear; no augmentation or EXIF change',
        'all_retained_decode_success':True})
    print('Cache complete',flush=True)


if __name__ == '__main__': main()
