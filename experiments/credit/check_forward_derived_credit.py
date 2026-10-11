"""B1/R1 reference-learner contracts and reused fresh per-future credit labels."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import time
import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode
from shared_carrier_paired_replay import replay_pairs
from reference_learner_route_credit import shadow_scores
from sparse_paired_actor_critic import initial,native,scores

ROOT=Path(__file__).resolve().parents[2]
SOURCE='experiments/results/credit/curie_future_packet_v2_20261011T0740Z_measure.json'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args()
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
    out=ROOT/f'experiments/results/credit/{a.tag}.json';assert not out.exists();start=time.monotonic()
    carrier,baseline,_,_,_=initial();checks=[]
    for depth in [2,4,8]:
        model=carrier if depth==4 else native.actor(depth,170);model.requires_grad_(False)
        t,m=native.base.batch(984000+depth,size=1,pairs=2);routes=[1,7];actions=native.base.actions()[[3,8]]
        actual,aw,ad=replay_pairs(model,t,m,routes,actions,.05,True)
        projected,pw,pd=shadow_scores(model,t,m,routes,actions,1,.05,True)
        errors=dict(reference_score=float((actual[1]-projected[1]).abs().max()),
            reference_weight=max(float((aw[k][1]-pw[k]).abs().max()) for k in pw),
            reference_gradient=max(float((u[k][1]-v[k]).abs().max()) for u,v in zip(ad['gradients'],pd['gradients']) for k in u),
            reference_adam=max(float((ad['optimizer']['state'][k][n][1]-pd['optimizer']['state'][k][n]).abs().max()) for k in ad['optimizer']['state'] for n in ['exp_avg','exp_avg_sq']))
        zero,_,zd=shadow_scores(model,t,m,routes,actions,1,0.,True)
        errors['zero_lr_static']=float((zero-zd['frozen_scores']).abs().max())
        suffix=m.clone();suffix[:,5:]=(suffix[:,5:]+1)%8
        other,_=shadow_scores(model,t,suffix,routes,actions,1,.05)
        errors['causal_first_score']=float((other[:,0]-projected[:,0]).abs().max())
        assert max(errors.values())<1e-11;checks.append(dict(depth=depth,errors=errors,
            alternative_projection_error=float((actual[0]-projected[0]).abs().max())))
    source=json.loads((ROOT/SOURCE).read_text());assert source['status']=='completed'
    for p,digest in source['source_sha256'].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==digest,p
    cp=ROOT/source['checkpoint'];assert hashlib.sha256(cp.read_bytes()).hexdigest()==source['checkpoint_sha256']
    saved=torch.load(cp,weights_only=False,map_location='cpu');truth=np.array(saved['dev_truth']);assert truth.shape==(32,8)
    predicted=dict(saved['dev_predictions']);static=[];shadow=[];unseen=[];tick=time.monotonic()
    routes=torch.arange(8).repeat_interleave(9).tolist();actions=native.base.actions().repeat(8,1)
    with FlopCounterMode(display=False) as counter:
        for i in range(32):
            t,m=native.base.batch(146823000+i,size=1,pairs=2)
            with torch.no_grad():x=native.prefix(carrier,t,m,3,list(range(8)));reference=int(scores(baseline,x,0).flatten().argmin())
            probe=int(np.random.default_rng(83000000+i).integers(72))
            if probe==reference or probe%9==reference%9==0:probe=(probe+1)%72
            unseen.append(probe//9>=6 or reference//9>=6);rng=np.random.default_rng(84000000+i);ss=[];bb=[]
            for _ in range(8):
                ft,fm=native.prior.continuation(t,m,2,rng)
                value,_,detail=shadow_scores(carrier,ft,fm,routes,actions,reference,.05,True)
                frozen=detail['frozen_scores'].mean(1);value=value.mean(1)
                ss.append(float(frozen[probe]-frozen[reference]));bb.append(float(value[probe]-value[reference]))
            static.append(ss);shadow.append(bb)
    predicted['forward_static']=static;predicted['reference_learner']=shadow
    predicted['shuffled_static']=np.roll(np.array(static),1,axis=1).tolist()
    weights=np.array(saved['dev_logit_importance_weights']);rows=[]
    for name,pred in predicted.items():
        error=(np.array(pred)-truth)**2
        rows.append(dict(arm=name,mse=float(error.mean()),error_over_zero=float(error.sum()/np.sum(truth**2)),
            unseen_probe_mse=float(error[np.array(unseen)].mean()) if any(unseen) else None,
            policy_logit_residual_second_moment=float((error*weights[:,None]).mean())))
    reference_rows={r['arm']:r for r in rows};comparisons={};indices=np.random.default_rng(861).integers(0,32,(4000,32))
    for first,second in [('forward_static','prefix_prior'),('reference_learner','prefix_prior'),('reference_learner','forward_static')]:
        gain=((np.array(predicted[second])-truth)**2-(np.array(predicted[first])-truth)**2).mean(1)
        comparisons[f'{first}_over_{second}']=dict(mse_gain=float(gain.mean()),prefix_bootstrap95=np.quantile(gain[indices].mean(1),[.025,.975]).tolist())
    best=min(['forward_static','reference_learner'],key=lambda n:reference_rows[n]['policy_logit_residual_second_moment'])
    gate=reference_rows[best]['mse']<reference_rows['prefix_prior']['mse'] and reference_rows[best]['policy_logit_residual_second_moment']<reference_rows['prefix_prior']['policy_logit_residual_second_moment'] and comparisons[f'{best}_over_prefix_prior']['prefix_bootstrap95'][0]>0
    result=dict(status='completed',battle='B1/R1',tag=a.tag,contracts=checks,rows=rows,comparisons=comparisons,
        candidate_predictor=best,forward_information_gate=gate,dev_prefixes=32,futures_per_prefix=8,
        projections_per_future=72,reference_readout_updates=1024,new_dev_label_teacher_worlds=0,
        diagnostic_supported_flops=counter.get_total_flops(),diagnostic_wall_s=time.monotonic()-tick,
        whole_wall_s=time.monotonic()-start,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        source_result=SOURCE,source_result_sha256=hashlib.sha256((ROOT/SOURCE).read_bytes()).hexdigest(),source_checkpoint_sha256=source['checkpoint_sha256'],
        source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['experiments/credit/check_forward_derived_credit.py','experiments/credit/reference_learner_route_credit.py','experiments/credit/shared_carrier_paired_replay.py',*native.SOURCES]},
        scope='Fixed native depth4 carrier, one saved actor seed170, same256 fresh per-future learning-aware contrast labels reused from completed future pilot. Learning-side predictors use revealed exogenous observations; actor not fitted and never receives future inputs. Exact frozen response and reference-learning projection, not learned persistent across-episode credit parameters. Shadow reference alone learns288 readout parameters from dense causal feedback; alternatives project its trajectory and need audit repair. All72 projected outputs and1024reference updates charged, inherited512 acquisition/earlier512teacherworlds/pretraining additional. Logit residual second moment is not native-parameter gradient variance or actor quality. Shuffled future response is a correspondence diagnostic. Contracts paid separately inside whole phase; diagnostic arithmetic omits unsupported/scalar/optimizer operations, inside wall. No TEST/general internal-route/depth-scaling or hardware claim.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
