"""B1/R1 matched causal controller learning using prefix/static/reference credit."""
import argparse
from copy import deepcopy
import hashlib,json,resource,time
import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode
import sparse_paired_actor_critic as old
from joint_forward_credit_episode import episode
from shared_carrier_paired_replay import replay_pairs
ROOT=old.ROOT
TAG='curie_joint_forward_credit_confirmation_20261011_v1'
SOURCES=['experiments/credit/confirm_joint_forward_credit.py','experiments/credit/joint_forward_credit_episode.py','experiments/credit/shared_carrier_paired_replay.py',*old.SOURCES]
ARMS=['prefix','forward_static','reference_learner']

def verify(path):
    d=json.loads((ROOT/path).read_text());assert d['status']=='completed'
    for p,digest in d['source_sha256'].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==digest,p
    return d

def train(state,carrier,baseline,tau,end,tag,replicate):
    start=time.monotonic();begin=state['cursor'];rows=[];worlds=updates=projections=0
    with FlopCounterMode(display=False) as work:
        for i in range(begin,end):
            t,m=old.native.base.batch(144701000+i+1000000*replicate,size=1,pairs=2)
            with torch.no_grad():
                x=old.native.prefix(carrier,t[:,:4],m[:,:4],3,list(range(6)))
                prefix=old.scores(baseline,x,0).detach();reference=int(prefix.flatten().argmin())
            pi=old.probabilities(old.scores(state['actor'],x,i),tau)
            draw=np.random.default_rng(73000000+i+1000000*replicate).random()
            action=min(int(np.searchsorted(pi.detach().flatten().numpy().cumsum(),draw)),53)
            ft,fm=old.native.prior.continuation(t,m,2,np.random.default_rng(75000000+i+1000000*replicate))
            if state['arm']=='prefix':
                ids=[action,reference]
                result=episode(carrier,ft,fm,[a//9 for a in ids],old.native.base.actions()[[a%9 for a in ids]],0,1,.05,False)
                credit=prefix
            else:
                ids=list(range(54))
                result=episode(carrier,ft,fm,[a//9 for a in ids],old.native.base.actions().repeat(6,1),action,reference,.05,state['arm']=='reference_learner')
                credit=result['projected_scores' if state['arm']=='reference_learner' else 'frozen_scores'].mean(1)
            contrast=float((result['actual_scores'][0]-result['actual_scores'][1]).mean())
            h=credit.detach().flatten()-credit.detach().flatten()[reference]
            loss=old.policy_loss(pi,credit,action,reference,True,contrast,1.)
            state['policy_opt'].zero_grad();loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(state['actor'].parameters(),1.);assert torch.isfinite(norm)
            state['policy_opt'].step();state['cursor']=i+1
            # Full correction is legal because this predictor/reference does not depend on sampled A.
            row=dict(prefix=i,action=action,reference=reference,factual_loss=float(result['actual_scores'][0].mean()),contrast=contrast,prediction=float(h[action]),residual_square=(contrast-float(h[action]))**2,actual_worlds=result['actual_worlds'])
            state['history'].append(row);rows.append(row)
            worlds+=result['actual_worlds'];updates+=result['readout_updates']
            projections+=result['projected_route_targets']
            if (i+1)%32==0 or i+1==end:
                preview=ROOT/'.git/joint-forward-credit-preview'/tag/state['arm'];preview.mkdir(parents=True,exist_ok=True)
                torch.save(old.pack(state),preview/'resume.pt')
                progress=dict(arm=state['arm'],cursor=i+1,wall_s=time.monotonic()-start,supported_flops=work.get_total_flops(),actual_worlds=worlds)
                (preview/'progress.json').write_text(json.dumps(progress,indent=2)+'\n');print(json.dumps(progress),flush=True)
    assert len(state['history'])==state['cursor']
    return dict(arm=state['arm'],start_prefix=begin,end_prefix=end,policy_updates=end-begin,factual_targets=(end-begin)*4,actual_worlds=worlds,readout_updates=updates,projected_route_targets=projections,supported_flops=work.get_total_flops(),wall_s=time.monotonic()-start,mean_factual_loss=float(np.mean([r['factual_loss'] for r in rows])),mean_sampled_residual_square=float(np.mean([r['residual_square'] for r in rows])))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);ap.add_argument('--mode',choices=['smoke','fit'],required=True);ap.add_argument('--replicate',type=int,choices=[1,2],required=True);a=ap.parse_args()
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64);start=time.monotonic()
    out=ROOT/f'experiments/results/credit/{a.tag}.json';assert not out.exists()
    verify('experiments/results/credit/curie_joint_forward_credit_20261011_v1_contract.json')
    if a.mode=='fit':verify(f'experiments/results/credit/{TAG}_smoke.json')
    carrier,baseline,tau,_,_=old.initial()
    prior,cp=old.load(f'experiments/results/credit/{old.TAG}_acquire.json')
    for p,digest in prior['source_sha256'].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==digest,p
    states={n:old.unpack(deepcopy(cp['states']['full_fixed']),baseline) for n in ARMS}
    for n,state in states.items():state['arm']=n
    for i,n in enumerate(ARMS):
        for other in ARMS[i+1:]:
            assert states[n]['history'] is not states[other]['history']
            for k,v in states[n]['policy_opt'].state_dict()['state'].items():
                for field in ['step','exp_avg','exp_avg_sq']:
                    assert v[field].data_ptr()!=states[other]['policy_opt'].state_dict()['state'][k][field].data_ptr()
    end=132 if a.mode=='smoke' else 512
    fits=[train(states[n],carrier,baseline,tau,end,a.tag,a.replicate) for n in ARMS]
    evaluation=None
    parent=next(f for f in prior['metrics']['fits'] if f['arm']=='full_fixed');cumulative=[]
    for fit in fits:
        cumulative.append(dict(arm=fit['arm'],factual_targets=parent['unique_factual_targets']+fit['factual_targets'],policy_updates=parent['policy_updates']+fit['policy_updates'],supported_flops=parent['supported_flops']+fit['supported_flops'],wall_s=parent['wall_s']+fit['wall_s']))
    payload=dict(states={n:old.pack(s) for n,s in states.items()},evaluation=evaluation)
    checkpoint=out.with_suffix('.pt');torch.save(payload,checkpoint)
    result=dict(status='completed',battle='B1/R1',tag=a.tag,mode=a.mode,replicate=a.replicate,fits=fits,cumulative_fits=cumulative,evaluation=evaluation,tau=tau,source_acquisition=f'experiments/results/credit/{old.TAG}_acquire.json',source_checkpoint_sha256=prior['checkpoint_sha256'],checkpoint=str(checkpoint.relative_to(ROOT)),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),whole_wall_s=time.monotonic()-start,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCES},scope='One-seed synthetic route-policy development. Identical valid128 full_fixed controller/Adam acquisition,384 continued causal episodes, same data/draws/rates. Fixed depth4 temporal/keyed carrier; independent288-parameter readout learners adapt4 times per episode. Prefix-only inference; static/reference credit sees arrived exogenous data only during learning. Full raw residual correction p1, nonlinear Adam/clipping not unbiased updates. All actual worlds/projected outputs/optimizer/copy costs in wall; supported FLOPs omit unsupported scalar/special/optimizer arithmetic. Original128 acquisition included per arm; source pretraining, diagnostic gates and fresh confirmation DEV is evaluated separately and charged; same128 acquisition is shared across optimization streams, not three actor initializations. No TEST/all-carrier/market continuity/hardware/depth-scaling claim.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
