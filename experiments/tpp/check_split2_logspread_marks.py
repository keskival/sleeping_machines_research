"""B4 TRAIN-only mark-branch contract against saved stable model and v20."""
import argparse
import hashlib
import json
import random
import resource
import time
from pathlib import Path
import numpy as np
import torch
import race_tpp_b4g as adapter
from race_tpp_logspread import RaceTPP as Stable
from race_tpp_logspread_marks import RaceTPP as Developed
from race_tpp_v20 import RaceTPP as Reference

ROOT=adapter.ROOT


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--tag',required=True);a=p.parse_args()
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64);torch.manual_seed(215);start=time.monotonic()
    pilot_path=ROOT/'experiments/results/tpp_b4/aws_b4_split2_logspread_pilot_20261011_v2.json'
    r=json.loads(pilot_path.read_text());assert r['status']=='completed' and r['skipped_updates']==0
    for s,h in r['source_sha256'].items():assert hashlib.sha256((ROOT/s).read_bytes()).hexdigest()==h
    train,K=adapter.load('github_filtered',2,'train');cfg=r['config']
    gaps=np.concatenate([np.diff(t) for t,_ in train]);pos=gaps[gaps>0]
    qs=np.log(np.quantile(pos,np.linspace(.1,.9,cfg['n_lognormal']))).tolist()
    args=(K+1,cfg['d'],cfg['modes'],cfg['layers'],cfg['n_exp'],cfg['n_lognormal'],cfg['dv'],cfg['dropout'],r['scale'],qs,r['recording_cell'],0,None,cfg['state_modes'])
    base=Stable(*args);new=Developed(*args);repair_path=ROOT/'experiments/results/tpp_b4/aws_b4_split2_logspread_develop_20261011.json';repair=json.loads(repair_path.read_text());assert repair['status']=='completed' and repair['skipped_updates']==0
    for source,digest in repair['source_sha256'].items():assert hashlib.sha256((ROOT/source).read_bytes()).hexdigest()==digest
    checkpoint=ROOT/repair['best_checkpoint'];state=torch.load(checkpoint,weights_only=True)
    base.load_state_dict(state);missing=new.load_state_dict(state,strict=False)
    assert not missing.unexpected_keys and all(k.startswith(('mark_layers.','mark_extra.')) for k in missing.missing_keys)
    for model in (base,new):model.eval()
    batch=next(adapter.base.batches([(t[:32],m[:32]) for t,m in train[:2]],2,False,random.Random(0)))
    t,m,mask=batch
    baseline=base.loglik(*batch)[:2];zero=new.loglik(*batch)[:2]
    zero_error=max(abs(float((x-y).detach())) for x,y in zip(baseline,zero));assert zero_error<1e-9
    with torch.no_grad():new.mark_extra.weight.normal_(0,.01)
    old=Reference(*args,mark_mem=1);old.load_state_dict(new.state_dict());old.eval()
    original=old.loglik(*batch)[:2];repaired=new.loglik(*batch)[:2]
    v20_error=max(abs(float((x-y).detach())) for x,y in zip(original,repaired));assert v20_error<1e-9
    tl,ml,_,_=new.loglik(*batch)
    time_grads=torch.autograd.grad(tl,list(new.mark_layers.parameters()),allow_unused=True,retain_graph=True)
    mark_grads=torch.autograd.grad(ml,list(new.mark_layers.parameters()),allow_unused=True)
    time_norm=sum(float(x.abs().sum()) for x in time_grads if x is not None)
    mark_norm=sum(float(x.abs().sum()) for x in mark_grads if x is not None)
    assert time_norm==0 and mark_norm>0
    new.zero_grad(set_to_none=True);tl,ml,count,_=new.loglik(*batch,r['recording_cell']);loss=-(tl+ml)/count;loss.backward()
    finite=bool(torch.isfinite(loss)) and all(bool(torch.isfinite(p.grad).all()) for p in new.parameters() if p.grad is not None);assert finite
    sources=['experiments/tpp/check_split2_logspread_marks.py','experiments/tpp/race_tpp_logspread_marks.py','experiments/tpp/race_tpp_logspread.py','experiments/tpp/race_tpp_v20.py','experiments/tpp/race_tpp_v19.py','experiments/tpp/race_tpp_b4g.py']
    result=dict(status='completed',tag=a.tag,battle='B4 development',stage='logspread_marks_contract',
                zero_branch_loss_error=zero_error,v20_mark_path_loss_error=v20_error,time_gradient_norm=time_norm,
                mark_gradient_norm=mark_norm,full_gradient_finite=finite,train_targets=int(count),
                parameters=sum(p.numel() for p in new.parameters()),base_parameters=sum(p.numel() for p in base.parameters()),
                source_sha256={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sources},
                checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                data_sha256=hashlib.sha256((adapter.DATA/'github_filtered'/'split_2'/'train.json').read_bytes()).hexdigest(),
                wall_s=time.monotonic()-start,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                scope='Two short TRAIN sequences, saved repaired checkpoint; zero-init timing/mark parity and established v20 branch parity. No TEST or quality claim.',
                decision='Passing admits fixed mark-memory TRAIN/DEV development fit; timing invariance is a contract, not a promised training outcome.')
    out=ROOT/'experiments/results/tpp_b4'/f'{a.tag}.json'
    with out.open('x') as stream:stream.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
