"""Portable exact B10 TRAIN/DEV windows; no test data or model dependency."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def export(source,bundle,manifest):
    bundle=Path(bundle);manifest=Path(manifest)
    if bundle.exists() or manifest.exists():raise FileExistsError('Unique immutable dataset artifacts required')
    fields={};record=dict(schema=1,window=1024,context=128,recording_cell_s=1e-6,splits={},sources={},scope='Exact existing B10 subsampled TRAIN/DEV windows only; no September TEST, no parent model. Prices/quantities retained; continuous-day volatility protocol requires additional data.')
    for split,first,days,every in [('train',dt.date(2026,8,1),24,20),('dev',dt.date(2026,8,25),7,10)]:
        lists={name:[] for name in ('t','mark','p','q')};dates=[];starts=[]
        for offset in range(days):
            day=str(first+dt.timedelta(offset));path=Path(source)/(day+'.npz')
            record['sources'][day]=sha(path)
            with np.load(path,allow_pickle=False) as raw:
                for start in range(0,len(raw['t'])//1024,every):
                    low=start*1024;high=low+1024
                    for name in lists:lists[name].append(raw[name][low:high].copy())
                    dates.append(day);starts.append(low)
        for name,values in lists.items():fields[f'{split}_{name}']=np.stack(values)
        fields[f'{split}_day']=np.array(dates);fields[f'{split}_start']=np.array(starts,dtype=np.int64)
        record['splits'][split]=dict(days=[str(first+dt.timedelta(i)) for i in range(days)],stride_windows=every,windows=len(dates),training_targets=len(dates)*1023,scored_targets=len(dates)*896)
    bundle.parent.mkdir(parents=True,exist_ok=True)
    with bundle.open('xb') as stream:np.savez_compressed(stream,**fields)
    record['bundle_sha256']=sha(bundle);record['bundle_bytes']=bundle.stat().st_size
    with manifest.open('x') as stream:json.dump(record,stream,indent=2);stream.write('\n')
    return record


def audit(bundle,manifest):
    record=json.loads(Path(manifest).read_text())
    if record['schema']!=1 or sha(bundle)!=record['bundle_sha256']:raise ValueError('Bundle identity mismatch')
    with np.load(bundle,allow_pickle=False) as arrays:
        for split,item in record['splits'].items():
            if split not in ('train','dev'):raise ValueError('Unexpected sealed split')
            t=arrays[f'{split}_t'];marks=arrays[f'{split}_mark']
            if t.dtype!=np.int64 or t.shape!=(item['windows'],1024) or marks.shape!=t.shape:raise ValueError('Window schema mismatch')
            if bool((np.diff(t,axis=1)<0).any()) or bool(((marks<0)|(marks>=6)).any()):raise ValueError('Invalid chronology/marks')
            if set(arrays[f'{split}_day'].tolist())!=set(item['days']):raise ValueError('Split day mismatch')
            if any(day>='2026-09-01' for day in item['days']):raise ValueError('Sealed TEST cannot enter this artifact')
            for name in ('p','q'):
                value=arrays[f'{split}_{name}']
                if value.shape!=t.shape or not bool(np.isfinite(value).all()):raise ValueError('Price/quantity schema mismatch')
    return record
