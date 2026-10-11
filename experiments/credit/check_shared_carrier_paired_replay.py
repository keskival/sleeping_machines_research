"""B1/R1 exact branch sharing and work scaling; no model fit or TEST."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import time
import torch
from torch.utils.flop_counter import FlopCounterMode
from cached_paired_native_replay import replay_pairs as previous
from shared_carrier_paired_replay import replay_pairs as shared
from sparse_paired_actor_critic import initial,native

ROOT=Path(__file__).resolve().parents[2]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args()
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
    out=ROOT/f'experiments/results/credit/{a.tag}.json';assert not out.exists();start=time.monotonic()
    trained,_,_,_,_=initial();rows=[]
    for depth in [2,4,8]:
        model=trained if depth==4 else native.actor(depth,170);model.requires_grad_(False)
        t,m=native.base.batch(973000+depth,size=1,pairs=2)
        for worlds in [2,18,72] if depth==4 else [2,18]:
            routes=torch.arange(8 if worlds==72 else 2).repeat_interleave(9 if worlds>2 else 1)
            if worlds<72:routes=torch.where(routes==0,torch.tensor(1),torch.tensor(7))
            ids=torch.arange(9).repeat(worlds//9) if worlds>2 else torch.tensor([3,8])
            actions=native.base.actions()[ids]
            old,ow,od=previous(model,t,m,routes.tolist(),actions,.05,True)
            new,nw,nd=shared(model,t,m,routes.tolist(),actions,.05,True)
            errors=dict(scores=float((old-new).abs().max()),weights=max(float((ow[k]-nw[k]).abs().max()) for k in ow),
                gradients=max(float((u[k]-v[k]).abs().max()) for u,v in zip(od['gradients'],nd['gradients']) for k in u),
                moments=max(float((od['optimizer']['state'][k][name]-nd['optimizer']['state'][k][name]).abs().max()) for k in od['optimizer']['state'] for name in ['step','exp_avg','exp_avg_sq']))
            frozen,_=previous(model,t,m,routes.tolist(),actions,0.)
            errors['frozen_scores']=float((frozen-nd['frozen_scores']).abs().max());assert max(errors.values())<1e-10
            suffix_t=t.clone();suffix_t[:,5:]+=1.;suffix_m=m.clone();suffix_m[:,5:]=(suffix_m[:,5:]+1)%8
            changed,_=shared(model,suffix_t,suffix_m,routes.tolist(),actions,.05)
            errors['suffix_first_score']=float((changed[:,0]-new[:,0]).abs().max());assert errors['suffix_first_score']<1e-12
            rows.append(dict(depth=depth,worlds=worlds,errors=errors,shared_carrier_values=nd['shared_carrier_values'],route_specific_values=nd['route_specific_values']))
    timing=[];t,m=native.base.batch(979999,size=1,pairs=2)
    for worlds in [2,18,72]:
        routes=torch.arange(8 if worlds==72 else 2).repeat_interleave(9 if worlds>2 else 1)
        ids=torch.arange(9).repeat(worlds//9) if worlds>2 else torch.tensor([3,8]);actions=native.base.actions()[ids]
        for name,fn in [('per_world_carrier',previous),('shared_carrier',shared)]:
            times=[];work=[]
            for _ in range(5):
                tick=time.monotonic()
                with FlopCounterMode(display=False) as counter:fn(trained,t,m,routes.tolist(),actions,.05)
                times.append(time.monotonic()-tick);work.append(counter.get_total_flops())
            timing.append(dict(worlds=worlds,arm=name,iterations=5,median_wall_s=sorted(times)[2],supported_flops_per_iteration=work[0],all_wall_s=times))
    result=dict(status='completed',battle='B1/R1',tag=a.tag,contracts=rows,timing=timing,
        mathematical_gate=True,whole_wall_s=time.monotonic()-start,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['experiments/credit/check_shared_carrier_paired_replay.py','experiments/credit/shared_carrier_paired_replay.py','experiments/credit/cached_paired_native_replay.py',*native.SOURCES]},
        scope='Fixed native timed/keyed carrier; optional key writes affect keyed output logits only. One causal carrier pass shared across specified counterfactuals; every world retains separate288-parameter readout/Adam coordinates and four supervised forecasts/updates. Scores/gradients/moments/weights/static response/suffix causality contracted at depths2/4/8. Same whole setup/cache/readout-learning boundary and five repetitions within each world-count timing comparison; supported arithmetic omits scalar/special/unsupported optimizer operations, inside wall. Carrier state does not scale with world count; route-specific bonuses/mark probabilities/readout state do. All eight marks are scored. Not changing-carrier, general internal-route, full-fit, learned-credit or clockless hardware evidence.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
