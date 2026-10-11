"""B1 replay-free full Taxi training; correct available cohort and preserve prior sources."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import time
import numpy as np
import torch
import train_native_future_update as base
from train_native_future_update_reuse import fit
ROOT=Path(__file__).resolve().parents[2]
SOURCES=['experiments/credit/train_native_future_update_reuse_full.py','experiments/credit/train_native_future_update_reuse.py',*base.SOURCES]


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--tag',required=True);ap.add_argument('--seed',type=int,default=0)
    ap.add_argument('--gate',required=True);a=ap.parse_args();a.stage='fit';a.epochs=3
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64);start=time.monotonic()
    path=ROOT/a.gate;gate=json.loads(path.read_text());assert gate['status']=='completed' and gate['stage']=='smoke'
    assert all(r['retained_replay_parameter_parity']<1e-10 for r in gate['rows'])
    for s,digest in gate['source_sha256'].items():assert hashlib.sha256((ROOT/s).read_bytes()).hexdigest()==digest
    raw=base.native.load_split('taxi','train');validation=base.native.load_split('taxi','dev')
    train=[(t[:32],m[:32]) for t,m in raw if len(m)>=32];assert len(train)==1400
    dev=[(t[:32],m[:32]) for t,m in validation if len(m)>=32][:128];assert len(dev)==128
    K=1+max(int(m.max()) for _,m in raw)
    gaps=np.concatenate([np.diff(t) for t,_ in train]);positive=gaps[gaps>0]
    scale=float(np.median(positive));quantiles=np.log(np.quantile(positive,[.1,.35,.65,.9])).tolist()
    rows=[];weights={}
    for depth in [2,4,8]:
        result,saved=fit(a,depth,train,dev,scale,quantiles,K);rows+=result;weights[str(depth)]=saved
        print(json.dumps(dict(completed_depth=depth,rows=result)),flush=True)
    folder=ROOT/'experiments/results/credit';checkpoint=folder/f'{a.tag}.pt';assert not checkpoint.exists()
    torch.save(weights,checkpoint)
    result=dict(status='completed',tag=a.tag,battle='B1 whole-native future-taught learning',stage='fit',rows=rows,
                args=vars(a),train_sequences=1400,events_per_sequence=32,dev_sequences=128,
                final_weights=str(checkpoint.relative_to(ROOT)),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                source_sha256={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in SOURCES},
                input_sha256={a.gate:hashlib.sha256(path.read_bytes()).hexdigest()},
                data_sha256={s:hashlib.sha256((ROOT/f'data/easytpp/taxi/{s}.json').read_bytes()).hexdigest() for s in ['train','dev']},
                wall_s=time.monotonic()-start,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                scope='Matched native full-parameter learning; depths2/4/8,all1400TRAIN sequences cropped32events,threepasses. Future-taught group scales reuse actual next actor gradient,zero extra forward replay. All controller/optimizer work counted in fit wall and supported FLOPs; unsupported operations are excluded from FLOP count. Dense writes/state reset across sequences; DEV only,one-seed development stage,no hard-route/full-horizon/Transformer or benchmark supremacy claim.')
    with (folder/f'{a.tag}.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))

if __name__=='__main__':main()
