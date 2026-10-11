"""B1/R1 paid exact route-controller gradient variance at fixed issued weights."""
import argparse,hashlib,json,resource,time
from copy import deepcopy
import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode
import sparse_paired_actor_critic as old
from joint_forward_credit_episode import episode
from shared_carrier_paired_replay import replay_pairs
ROOT=old.ROOT
SOURCES=['experiments/credit/measure_forward_credit_variance.py','experiments/credit/joint_forward_credit_episode.py','experiments/credit/shared_carrier_paired_replay.py',*old.SOURCES]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args()
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64);start=time.monotonic()
    out=ROOT/f'experiments/results/credit/{a.tag}.json';assert not out.exists()
    carrier,baseline,tau,_,_=old.initial();prior,cp=old.load(f'experiments/results/credit/{old.TAG}_acquire.json')
    actor=deepcopy(baseline);actor.load_state_dict(cp['states']['full_fixed']['actor']);params=tuple(actor.parameters())
    names=['zero','prefix','forward_static','reference_learner'];rows=[];max_mean_error=max_autograd_error=0.;worlds=updates=0
    prefix_count,future_count=12,16;ids=list(range(54));messages=old.native.base.actions().repeat(6,1)
    with FlopCounterMode(display=False) as work:
        for i in range(prefix_count):
            # Diagnostic uses TRAIN-domain observations and all TRAIN routes; no DEV/TEST selection.
            t,m=old.native.base.batch(144701512+i,size=1,pairs=2)
            with torch.no_grad():
                x=old.native.prefix(carrier,t[:,:4],m[:,:4],3,list(range(6)))
                prefix=old.scores(baseline,x,0).flatten().detach();reference=int(prefix.argmin())
            pi=old.probabilities(old.scores(actor,x,128),tau).flatten();prob=pi.detach()
            score=torch.stack([torch.cat([v.flatten() for v in torch.autograd.grad(pi[k].log(),params,retain_graph=True)]) for k in ids]).detach()
            conditional={n:[] for n in names};means=[];squared_error={n:[] for n in names};rng=np.random.default_rng(88000000+i)
            for f in range(future_count):
                ft,fm=old.native.prior.continuation(t,m,2,rng)
                true,_=replay_pairs(carrier,ft,fm,[k//9 for k in ids],messages,.05)
                projected=episode(carrier,ft,fm,[k//9 for k in ids],messages,reference,reference,.05,True)
                worlds+=54+projected['actual_worlds'];updates+=54*4+projected['readout_updates']
                d=true.mean(1);d=d-d[reference]
                exact=(prob[:,None]*d[:,None]*score).sum(0);means.append(exact)
                gradient=torch.cat([v.flatten() for v in torch.autograd.grad((pi*d.detach()).sum(),params,retain_graph=True)])
                max_autograd_error=max(max_autograd_error,float((gradient-exact).abs().max()))
                predictions=dict(zero=torch.zeros_like(d),prefix=prefix,forward_static=projected['frozen_scores'].mean(1),reference_learner=projected['projected_scores'].mean(1))
                for n,h in predictions.items():
                    h=h-h[reference];g_h=(prob[:,None]*h[:,None]*score).sum(0)
                    g=g_h[None]+(d-h)[:,None]*score
                    expected=(prob[:,None]*g).sum(0)
                    max_mean_error=max(max_mean_error,float((expected-exact).abs().max()))
                    conditional[n].append(float((prob[:,None]*(g-exact[None]).square()).sum()))
                    squared_error[n].append(float((prob*(d-h).square()).sum()))
            means=torch.stack(means);future_variance=float((means-means.mean(0)).square().sum()/(future_count-1))
            for n in names:
                value=float(np.mean(conditional[n]));rows.append(dict(prefix=i,arm=n,conditional_route_gradient_variance=value,future_gradient_variance=future_variance,total_variance_estimate=value+future_variance,route_weighted_contrast_mse=float(np.mean(squared_error[n]))))
            print(json.dumps(dict(prefixes=i+1,wall_s=time.monotonic()-start,teacher_worlds=worlds)),flush=True)
    assert max(max_mean_error,max_autograd_error)<1e-9
    aggregate=[]
    for n in names:
        data=[r for r in rows if r['arm']==n]
        aggregate.append(dict(arm=n,**{field:float(np.mean([r[field] for r in data])) for field in ['conditional_route_gradient_variance','future_gradient_variance','total_variance_estimate','route_weighted_contrast_mse']}))
    result=dict(status='completed',battle='B1/R1',tag=a.tag,rows=aggregate,per_prefix=rows,prefixes=prefix_count,futures_per_prefix=future_count,routes_per_future=54,teacher_worlds=worlds,readout_updates=updates,raw_mean_error=max_mean_error,policy_autograd_error=max_autograd_error,controller_parameters=sum(v.numel() for v in params),supported_flops=work.get_total_flops(),whole_wall_s=time.monotonic()-start,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,source_acquisition=f'experiments/results/credit/{old.TAG}_acquire.json',source_checkpoint_sha256=prior['checkpoint_sha256'],source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCES},scope='Fixed issued10584-parameter controller from valid128 full_fixed checkpoint,12 TRAIN-domain prefixes/16 exogenous futures each. Exact probability-weighted action enumeration includes54 independently learning actual worlds plus one paid projected-reference world per future. Full p1 correction, no policy/critic update. Conditional route covariance trace computed exactly on finite route support; future covariance sample estimate uses16 futures. Genuine route-controller parameter gradients, not fixed native carrier parameter gradients. All teacher worlds, reference projections, Jacobian extraction and inherited work charged; supported FLOPs omit unsupported scalar/special/optimizer work, included in wall. No DEV/TEST, forward-quality, all-carrier/depth/hardware claim.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
