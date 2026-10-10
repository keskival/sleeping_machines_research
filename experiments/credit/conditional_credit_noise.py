"""R1/B1 conditional continuation noise and exact cached counterfactual values.
No fitting or TEST. Factual native queries/clocks are reused only because the
admitted key-write intervention does not affect their evolution on fixed inputs.
General internal routing does not satisfy that assumption automatically.
"""
import argparse
import hashlib
import json
import math
import resource
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode
import future_route_critic_v1 as base

SOURCES=['experiments/credit/conditional_credit_noise.py',*base.SOURCES]


def analytic_advantages(model,t,m,index,candidates):
    _,cache=base.event_losses(model,t,m)
    q=model._query[:,:-1]
    age=(t[:,:-1]-t[:,index:index+1]).clamp_min(0)/model.scale
    transport=torch.exp(-age[:,:,None]*model.key_log_rate.exp())
    beta=torch.einsum('bld,ad->bla',q*transport,candidates)/math.sqrt(model.dk)
    active=torch.arange(q.shape[1])[None,:,None]>=index
    beta=beta*active
    source=F.one_hot(m[:,index],model.K).to(q.dtype)
    logits=cache[2][4][:,:,:,None,:]+beta[:,:,None,:,None]*source[:,None,None,None,:]
    logpk=F.log_softmax(logits,-1)
    targets=m[:,1:,None,None,None].expand(-1,-1,model.M,len(candidates),1)
    selected=logpk.gather(-1,targets).squeeze(-1)
    lh,_=model.clock_terms(cache[3],*cache[2][:4])
    changed=torch.logsumexp(lh[:,:,:,None]+selected,2)
    original_selected=cache[2][4].gather(-1,m[:,1:,None,None].expand(-1,-1,model.M,1)).squeeze(-1)
    original=torch.logsumexp(lh+original_selected,2)
    return (original[:,:,None]-changed)[:,index:].mean(1)


def continuation(t,m,pairs,rng):
    # Observable prefix key/value pairs; independent queries with replacement
    # and the original MQAR gap laws. No hidden identity or regime label.
    tt=t[:,:2*pairs].tolist()[0];mm=m[:,:2*pairs].tolist()[0]
    keys=mm[::2];values=mm[1::2];now=tt[-1]
    for _ in range(pairs):
        p=int(rng.integers(0,pairs));now+=math.exp(rng.uniform(math.log(.05),math.log(50.)))
        tt.append(now);mm.append(keys[p]);now+=math.exp(rng.normal(math.log(.2),.5))
        tt.append(now);mm.append(values[p])
    return torch.tensor([tt],dtype=torch.float64),torch.tensor([mm],dtype=torch.long)


def contracts():
    rows=[]
    for depth in (2,4,8):
        model=base.actor(depth,174);t,m=base.batch(174,size=1,pairs=2);j=3
        candidates=torch.cat((base.actions(),torch.tensor([[.07,-.11,.05,.03]],dtype=torch.float64)),0)
        exact=analytic_advantages(model,t,m,j,candidates)
        original,_=base.event_losses(model,t,m);reference=[]
        for a in candidates:
            changed,_=base.event_losses(model,t,m,j,a[None,:])
            reference.append((changed[:,j:]-original[:,j:]).mean())
        reference=torch.stack(reference)[None,:]
        error=float((exact-reference).abs().max());assert error<1e-11,error
        assert abs(float(exact[0,0]))<1e-12
        rows.append(dict(depth=depth,maximum_branch_parity_error=error,candidates=len(candidates),
                         analytic_actor_forwards=1,reference_actor_forwards=len(candidates)+1,
                         scope='Shared factual queries and time-clock parameters; intervention changes only addressed key score, not future internal state or external inputs.'))
    return rows


def measure(a):
    model=base.actor(a.depth,a.seed);candidates=base.actions();A=[];B=[];raw=[];within=[];saved=[]
    with torch.no_grad():
        for i in range(a.prefixes):
            t,m=base.batch(300000+a.seed*1000+i,size=1,pairs=a.pairs);j=2*a.pairs-1
            packet=base.prefix_packet(model,t,m,j)
            halves=[]
            for half in (0,1):
                rng=np.random.default_rng(500000+a.seed*10000+i*100+half)
                values=[]
                for k in range(a.rollouts):
                    ft,fm=continuation(t,m,a.pairs,rng)
                    assert torch.equal(packet,base.prefix_packet(model,ft,fm,j))
                    values.append(analytic_advantages(model,ft,fm,j,candidates)[0])
                halves.append(torch.stack(values))
            all_values=torch.cat(halves);A.append(halves[0].mean(0));B.append(halves[1].mean(0))
            raw.append(all_values.square().sum(-1).mean())
            within.append(all_values.var(0,unbiased=True).sum())
            saved.append(dict(prefix=i,mean_advantage=all_values.mean(0).tolist(),
                              continuation_variance=all_values.var(0,unbiased=True).tolist()))
    x=torch.stack(A);y=torch.stack(B);N=len(x)
    cross=(x*y).sum(-1);conditional_energy=cross.mean()
    global_energy=((x.sum(0)*y.sum(0)).sum()-cross.sum())/(N*(N-1))
    signal=conditional_energy-global_energy
    total=float(torch.stack(raw).mean())
    return dict(prefixes=N,independent_rollouts_per_half=a.rollouts,
        conditional_mean_squared_norm=float(conditional_energy),
        global_mean_squared_norm=float(global_energy),
        context_signal_energy=float(signal),context_signal_fraction_of_raw_energy=float(signal)/max(total,1e-30),
        mean_conditional_noise_energy=float(torch.stack(within).mean()),raw_energy=total,
        per_prefix=saved,actor_forwards=N*4*a.rollouts+N,cached_response_forwards=N*2*a.rollouts,causal_prefix_checks=N*2*a.rollouts,
        scored_target_presentations=N*2*a.rollouts*((4*a.pairs-1)+(2*a.pairs-1))+N*(2*a.pairs-1),
        scope='Independent continuation halves give an unbiased conditional-mean energy witness; cross-prefix U-statistic estimates global mean energy. Finite estimates can be negative and are not clipped. Conditioning is the full observed synthetic prefix; sufficiency of current critic features is not assumed. Not a critic fit, scaling win or deployment simulator.')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True)
    ap.add_argument('--mode',choices=['contract','smoke','measure'],required=True)
    ap.add_argument('--seed',type=int,default=170);ap.add_argument('--depth',type=int,default=2)
    ap.add_argument('--pairs',type=int,default=2);ap.add_argument('--prefixes',type=int,default=16)
    ap.add_argument('--rollouts',type=int,default=8);a=ap.parse_args()
    out=base.ROOT/'experiments/results/credit'/(a.tag+'.json')
    if out.exists():raise FileExistsError(out)
    if a.mode=='smoke':a.prefixes=2;a.rollouts=2
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64);start=time.monotonic()
    with FlopCounterMode(display=False) as counter:result=contracts() if a.mode=='contract' else measure(a)
    record=dict(status='completed',tag=a.tag,battle='R1/B1',training=False,args=vars(a),metrics=result,
        decision='Is prefix-dependent future credit obscured by continuation noise, and can exact cached response replace full counterfactual replay?',
        wall_s=time.monotonic()-start,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        supported_flops=counter.get_total_flops(),flop_scope='Supported PyTorch formulas; includes all actors, analytic responses and prefix checks, excludes special/unsupported arithmetic.',
        source_sha256={s:hashlib.sha256((base.ROOT/s).read_bytes()).hexdigest() for s in SOURCES})
    temporary=out.with_suffix('.tmp');temporary.write_text(json.dumps(record,indent=2)+'\n');temporary.replace(out)
    print('RESULT',json.dumps({k:v for k,v in record.items() if k!='metrics'}),flush=True)


if __name__=='__main__':main()
