"""B1/R1 numerical and measured-work gate for fixed-carrier replay hoisting."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import resource
import time
import torch
from torch.utils.flop_counter import FlopCounterMode
from optimizer_aware_route_credit import BranchReadout, native
from paired_native_replay import replay_pairs as original
from cached_paired_native_replay import replay_pairs as cached
from sparse_paired_actor_critic import initial

ROOT=Path(__file__).resolve().parents[2]


def reference(source,t,m,routes,actions,lr):
    model=native.actor(len(source.layers),0);model.load_state_dict(source.state_dict());model.requires_grad_(False)
    model.mark_ctx=BranchReadout(model.mark_ctx,len(routes));opt=torch.optim.Adam(model.mark_ctx.parameters(),lr=lr)
    tt,mm=t.repeat(len(routes),1),m.repeat(len(routes),1);values=[];gradients=[]
    for target in range(4,m.shape[1]):
        opt.zero_grad();loss,_=native.losses(model,tt[:,:target+1],mm[:,:target+1],3,torch.tensor(routes),actions)
        current=loss[:,-1];values.append(current.detach());current.sum().backward()
        gradients.append({k:p.grad.detach().clone() for k,p in model.mark_ctx.named_parameters()});opt.step()
    return torch.stack(values,1),model.mark_ctx.state_dict(),dict(gradients=gradients,optimizer=opt.state_dict())


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args()
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
    out=ROOT/f'experiments/results/credit/{a.tag}.json';assert not out.exists();start=time.monotonic();rows=[]
    trained,_,_,_,_=initial()
    for depth in [2,4,8]:
        model=trained if depth==4 else native.actor(depth,170)
        model.requires_grad_(False);t,m=native.base.batch(930000+depth,size=1,pairs=2)
        for routes,ids,lr in [([0],[0],0.),([1,7],[3,8],.05)]:
            actions=native.base.actions()[ids]
            old,old_weights,old_detail=reference(model,t,m,routes,actions,lr)
            new,new_weights,new_detail=cached(model,t,m,routes,actions,lr,True)
            score_error=float((old-new).abs().max())
            weight_error=max(float((old_weights[k]-new_weights[k]).abs().max()) for k in old_weights)
            gradient_error=max(float((u[k]-v[k]).abs().max()) for u,v in zip(old_detail['gradients'],new_detail['gradients']) for k in u)
            moment_error=max(float((old_detail['optimizer']['state'][k][name]-new_detail['optimizer']['state'][k][name]).abs().max()) for k in old_detail['optimizer']['state'] for name in ['step','exp_avg','exp_avg_sq'])
            assert max(score_error,weight_error,gradient_error,moment_error)<1e-11
            # Changing unrevealed marks/gaps cannot change the first forecast.
            changed_t=t.clone();changed_t[:,5:]+=1.;changed_m=m.clone();changed_m[:,5:]=(changed_m[:,5:]+1)%8
            changed,_,_=cached(model,changed_t,changed_m,routes,actions,lr,True)
            causal_error=float((changed[:,0]-new[:,0]).abs().max());assert causal_error<1e-12
            rows.append(dict(depth=depth,worlds=len(routes),lr=lr,score_error=score_error,weight_error=weight_error,
                gradient_error=gradient_error,adam_state_error=moment_error,future_suffix_first_score_error=causal_error,
                retained_carrier_values=new_detail['retained_carrier_values']))
    t,m=native.base.batch(939999,size=1,pairs=2);actions=native.base.actions()[[3,8]];timing=[]
    for name,fn in [('original',original),('cached',cached)]:
        times=[];works=[]
        for _ in range(5):
            begin=time.monotonic()
            with FlopCounterMode(display=False) as counter:fn(trained,t,m,[1,7],actions,.05)
            times.append(time.monotonic()-begin);works.append(counter.get_total_flops())
        timing.append(dict(arm=name,iterations=5,worlds_per_iteration=2,forecasts_per_world=4,
            median_wall_s=sorted(times)[2],supported_flops_per_iteration=works[0],all_wall_s=times))
    ratio=timing[0]['median_wall_s']/timing[1]['median_wall_s']
    result=dict(status='completed',battle='B1/R1',tag=a.tag,contracts=rows,timing=timing,median_speedup=ratio,
        arithmetic_ratio=timing[1]['supported_flops_per_iteration']/timing[0]['supported_flops_per_iteration'],
        kernel_admission=ratio>=1.5,whole_wall_s=time.monotonic()-start,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['experiments/credit/check_cached_paired_native_replay.py','experiments/credit/cached_paired_native_replay.py','experiments/credit/paired_native_replay.py','experiments/credit/optimizer_aware_route_credit.py',*native.SOURCES]},
        scope='Exact fixed-carrier branch replay at depths2/4/8; independent causal readout Adam, all forecast/gradient/moment/weight checks. Depth4 uses saved trained carrier. Both timing rows are whole two-world/four-forecast replay with setup/copy/readout updates and FlopCounter; unsupported arithmetic omitted from FLOPs, included in wall. Future carrier features computed in one offline causal pass; suffix contract checks first forecast. This is kernel work, not full fit, learned-credit utility, changing-carrier or clockless hardware efficiency.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
