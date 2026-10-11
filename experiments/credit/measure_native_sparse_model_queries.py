"""B1/R1 TRAIN-only native sparse model-query work versus gradient variance."""
import argparse,hashlib,json,resource,time
from copy import deepcopy
import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode
import sparse_paired_actor_critic as old
from joint_forward_credit_episode import episode
from shared_carrier_paired_replay import replay_pairs
ROOT=old.ROOT
SOURCES=['experiments/credit/measure_native_sparse_model_queries.py','experiments/credit/joint_forward_credit_episode.py','experiments/credit/shared_carrier_paired_replay.py',*old.SOURCES]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args()
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64);start=time.monotonic()
    out=ROOT/f'experiments/results/credit/{a.tag}.json';assert not out.exists()
    gate=json.loads((ROOT/'experiments/results/credit/curie_sparse_model_term_20261011_v1_contract.json').read_text());assert gate['status']=='completed' and gate['gate']
    for p,digest in gate['source_sha256'].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==digest,p
    confirmed=json.loads((ROOT/'experiments/results/credit/curie_joint_forward_credit_confirmation_20261011_v1_evaluate.json').read_text());assert confirmed['status']=='completed' and confirmed['confirmation_gate']
    carrier,baseline,tau,_,_=old.initial();prior,cp=old.load(f'experiments/results/credit/{old.TAG}_acquire.json')
    actor=deepcopy(baseline);actor.load_state_dict(cp['states']['full_fixed']['actor']);params=tuple(actor.parameters())
    routes=list(range(54));messages=old.native.base.actions().repeat(6,1);calibration=[];worlds=updates=0
    with FlopCounterMode(display=False) as work:
        for i in range(4):
            t,m=old.native.base.batch(144702000+i,size=1,pairs=2)
            with torch.no_grad():
                x=old.native.prefix(carrier,t[:,:4],m[:,:4],3,list(range(6)))
                reference=int(old.scores(baseline,x,0).flatten().argmin())
            graph=old.probabilities(old.scores(actor,x,128),tau).flatten();pi=graph.detach()
            score=torch.stack([torch.cat([v.flatten() for v in torch.autograd.grad(graph[k].log(),params,retain_graph=True)]) for k in routes]).detach()
            rng=np.random.default_rng(92000000+i);gradients=[];model_var=[];residual_var=[]
            for _ in range(8):
                ft,fm=old.native.prior.continuation(t,m,2,rng)
                true,_=replay_pairs(carrier,ft,fm,[k//9 for k in routes],messages,.05)
                projected=episode(carrier,ft,fm,[k//9 for k in routes],messages,reference,reference,.05,True)
                worlds+=55;updates+=55*4
                d=true.mean(1);d=d-d[reference];h=projected['projected_scores'].mean(1);h=h-h[reference]
                gh=(pi[:,None]*h[:,None]*score).sum(0);gr=(pi[:,None]*(d-h)[:,None]*score).sum(0)
                gradients.append((pi[:,None]*d[:,None]*score).sum(0))
                model_var.append(float((pi[:,None]*(h[:,None]*score-gh).square()).sum()))
                residual_var.append(float((pi[:,None]*((d-h)[:,None]*score-gr).square()).sum()))
            gs=torch.stack(gradients);future=float((gs-gs.mean(0)).square().sum()/7)
            calibration.append(dict(prefix=i,future_variance=future,model_query_variance=float(np.mean(model_var)),residual_variance=float(np.mean(residual_var))))
    calibration_flops=work.get_total_flops();timing=[];reference_errors=[]
    # Identical final calibration episode and actual action across query budgets.
    draw=np.random.default_rng(93000000).random();action=min(int(np.searchsorted(pi.numpy().cumsum(),draw)),53)
    def operational(query_count,repeat):
        actor.zero_grad()
        with torch.no_grad():
            xx=old.native.prefix(carrier,t[:,:4],m[:,:4],3,list(range(6)))
            b=int(old.scores(baseline,xx,0).flatten().argmin())
        probabilities=old.probabilities(old.scores(actor,xx,128),tau).flatten()
        if query_count is None:ids=list(range(54));sampled=None
        else:
            sampled=np.random.default_rng(94000000+repeat).choice(54,size=query_count,p=probabilities.detach().numpy()).tolist()
            ids=list(dict.fromkeys([b,action,*sampled]))
        positions={k:j for j,k in enumerate(ids)}
        packet=episode(carrier,ft,fm,[k//9 for k in ids],old.native.base.actions()[[k%9 for k in ids]],positions[action],positions[b],.05,True)
        hh=packet['projected_scores'].mean(1);hh=hh-hh[positions[b]]
        reference_errors.append(float((hh-h[ids]).abs().max()))
        delta=float((packet['actual_scores'][0]-packet['actual_scores'][1]).mean())
        if sampled is None:loss=(probabilities*hh).sum()
        else:loss=(probabilities[sampled].log()*hh[[positions[k] for k in sampled]]).mean()
        loss=loss+(delta-hh[positions[action]])*probabilities[action].log();loss.backward()
        assert all(v.grad is None or torch.isfinite(v.grad).all() for v in params)
        return packet['actual_worlds'],len(ids)
    for q in [1,2,4,8,None]:
        operational(q,0);times=[];flops=[];counts=[]
        for r in range(5):
            tick=time.monotonic()
            with FlopCounterMode(display=False) as counter:count,queried=operational(q,r)
            times.append(time.monotonic()-tick);flops.append(counter.get_total_flops());counts.append(queried);worlds+=count;updates+=count*4
        timing.append(dict(arm='exact_all_routes' if q is None else f'sampled_{q}',queries=q,median_wall_s=float(np.median(times)),supported_flops_per_iteration=float(np.mean(flops)),distinct_projected_routes=counts,repetitions=5))
    assert max(reference_errors)<1e-11
    future=float(np.mean([v['future_variance'] for v in calibration]));model=float(np.mean([v['model_query_variance'] for v in calibration]));residual=float(np.mean([v['residual_variance'] for v in calibration]))
    for row in timing:
        variance=future+residual+(model/row['queries'] if row['queries'] else 0)
        row.update(total_gradient_variance_estimate=variance,iid_wall_variance_product=row['median_wall_s']*variance)
    best=min(timing,key=lambda row:row['iid_wall_variance_product'])['arm']
    result=dict(status='completed',battle='B1/R1',tag=a.tag,rows=timing,calibration=calibration,train_prefixes=4,futures_per_prefix=8,calibration_supported_flops=calibration_flops,calibration_teacher_worlds=4*8*55,measured_worlds_excluding_warmups=worlds,readout_updates_excluding_warmups=updates,projection_parity_error=max(reference_errors),preferred_measured_budget=best,whole_wall_s=time.monotonic()-start,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,source_checkpoint_sha256=prior['checkpoint_sha256'],source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCES},scope='Fixed acquired128 controller/native depth4 carrier, TRAIN-only4prefixes/8futures. Full54-route independently learning losses plus reference projections calibrate conditional residual/model sampling variance; future covariance sample estimate. Operational comparison includes prefix discovery/all54 policy normalization, factual/reference head copies/four causal Adam updates, candidate projection and actor backward; actor optimizer step omitted equally because weights stay fixed. Independent model queries use issued pi, with replacement; duplicates reuse projections. Five warmups additional to counted operational worlds, included in whole wall. Exact all-route contraction is distinct from iid54-query sampling. IID wall-times-variance product is a small calibration criterion, not Adam convergence, nonstationary causal throughput or hardware/depth claim. Complete wall includes unsupported/scalar/special/optimizer arithmetic; supported FLOPs do not. No DEV/TEST/fitting.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
