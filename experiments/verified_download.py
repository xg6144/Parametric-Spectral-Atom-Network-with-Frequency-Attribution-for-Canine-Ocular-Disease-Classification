"""Recover an interrupted official MedMNIST release with strict range/checksum checks."""
import argparse
import hashlib
import re
import shutil

import requests
from medmnist.info import INFO

from .common import ROOT,sha256,write_json


def download(name,size):
    info=INFO[name];url=info['url_'+str(size)];expected=info['MD5_'+str(size)]
    root=ROOT/'results/external_data';path=root/f'{name}_{size}.npz'
    if path.exists() and hashlib.md5(path.read_bytes()).hexdigest()==expected:
        print('Already checksum verified',path,flush=True);return
    record=re.search(r'/records/(\d+)/',url).group(1)
    response=requests.get('https://zenodo.org/api/records/'+record,timeout=(10,30));response.raise_for_status()
    item=next(f for f in response.json()['files'] if f['key']==path.name)
    if item['checksum']!='md5:'+expected:raise ValueError('Package/API checksum disagreement')
    total=int(item['size']);partial=path.with_suffix('.npz.partial')
    before=None
    if path.exists():
        before=dict(size=path.stat().st_size,md5=hashlib.md5(path.read_bytes()).hexdigest(),sha256=sha256(path))
        failed=root/'failed_downloads'/f'{path.name}.{before["sha256"][:12]}.incomplete'
        failed.parent.mkdir(parents=True,exist_ok=True)
        if not failed.exists():shutil.copy2(path,failed)
        before['preserved_at']=str(failed.relative_to(ROOT))
        if not partial.exists() and path.stat().st_size<total:shutil.copy2(path,partial)
    offset=partial.stat().st_size if partial.exists() else 0
    if offset>=total:
        # A complete-sized invalid prefix cannot be repaired by an append.
        partial.rename(partial.with_suffix('.invalid'));offset=0
    headers={'Range':f'bytes={offset}-{total-1}','Cache-Control':'no-cache'}
    stream=requests.get(item['links']['self']+f'?revision_range_start={offset}',headers=headers,
                        stream=True,timeout=(10,30));stream.raise_for_status()
    actual_range=stream.headers.get('Content-Range','')
    if stream.status_code!=206 or actual_range!=f'bytes {offset}-{total-1}/{total}':
        stream.close();raise ValueError('Server did not honor the exact required byte range: '+actual_range)
    print('Official byte-range recovery',offset,total,flush=True)
    with partial.open('ab' if offset else 'wb') as out:
        for chunk in stream.iter_content(1024*1024):out.write(chunk)
    stream.close()
    actual=hashlib.md5(partial.read_bytes()).hexdigest()
    if partial.stat().st_size!=total or actual!=expected:
        write_json(root/(name+str(size)+'_download_recovery.json'),dict(status='FAILED',before=before,
                   actual_size=partial.stat().st_size,expected_size=total,actual_md5=actual,expected_md5=expected))
        raise ValueError('Recovered file failed published size/MD5')
    partial.replace(path)
    write_json(root/(name+str(size)+'_download_recovery.json'),dict(status='COMPLETED',before=before,
        requested_offset=offset,downloaded_bytes=total-offset,expected_size=total,actual_size=total,
        actual_md5=actual,expected_md5=expected,source=url,api_file_metadata=item,
        source_code_sha256=sha256(__file__)))
    print('Recovered official file; published MD5 verified',path,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--name',required=True);parser.add_argument('--size',type=int,default=64)
    args=parser.parse_args();download(args.name,args.size)
