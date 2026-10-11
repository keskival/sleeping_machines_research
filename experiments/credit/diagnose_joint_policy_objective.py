"""B1/R1 exact greedy versus stochastic policy utility on completed DEV cohort."""
import argparse,hashlib,json,resource,time
from copy import deepcopy
import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode
import sparse_paired_actor_critic as old
from shared_carrier_paired_replay import replay_pairs
ROOT=old.ROOT
SOURCE='experiments/results/credit/curie_joint_forward_credit_20261011_v1_fit.json'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args()
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64);start=time.monotonic()
    out=ROOT/f'experiments/results/credit/{a.tag}.json';assert not out.exists()
    source,cp=old.load(SOURCE)
    for p,digest in source['source_sha256'].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==digest,p
    carrier,baseline,tau,_,_=old.initial();prior,acquired=old.load(source['source_acquisition'])
    actors={'baseline':deepcopy(baseline),'acquired128':deepcopy(baseline)}
    actors['acquired128'].load_state_dict(acquired['states']['full_fixed']['actor'])
    for name,payload in cp['states'].items():
        actors[name]=deepcopy(baseline);actors[name].load_state_dict(payload['actor'])
    names=list(actors);per_prefix=[];truth=[];all_pi={n:[] for n in names};max_replay_error=0.
    routes=torch.arange(8).repeat_interleave(9).tolist();actions=old.native.base.actions().repeat(8,1)
    with FlopCounterMode(display=False) as work:
        for i in range(64):
            t,m=old.native.base.batch(147913000+i,size=1,pairs=2)
            with torch.no_grad():
                x=old.native.prefix(carrier,t[:,:4],m[:,:4],3,list(range(8)))
                predictions={n:old.scores(net,x,512 if n in cp['states'] else 128).flatten() for n,net in actors.items()}
                probabilities={n:old.probabilities(v.reshape(1,8,9),tau).flatten() for n,v in predictions.items()}
                selected={n:int(v.argmin()) for n,v in predictions.items()}
            rng=np.random.default_rng(87000000+i);values=[]
            for _ in range(8):
                ft,fm=old.native.prior.continuation(t,m,2,rng)
                losses,_=replay_pairs(carrier,ft,fm,routes,actions,.05);values.append(losses.mean(1))
            q=torch.stack(values);truth.append(q);mean=q.mean(0)
            reference=selected['baseline'];base_expected=float((probabilities['baseline']*mean).sum())
            for n in names:
                pi=probabilities[n];all_pi[n].append(pi)
                greedy=float(mean[reference]-mean[selected[n]])
                if n in cp['states']:max_replay_error=max(max_replay_error,abs(greedy-source['evaluation']['prefix_gains'][n][i]))
                per_prefix.append(dict(prefix=i,arm=n,greedy_gain_vs_baseline=greedy,stochastic_gain_vs_baseline=base_expected-float((pi*mean).sum()),stochastic_gain_vs_acquired=float(((probabilities['acquired128']-pi)*mean).sum()),entropy=float(-(pi*pi.log()).sum()),kl_vs_acquired=float((pi*(pi.log()-probabilities['acquired128'].log())).sum()),unseen_route_probability=float(pi[54:].sum())))
            if (i+1)%16==0:print(json.dumps(dict(prefixes=i+1,wall_s=time.monotonic()-start)),flush=True)
    assert max_replay_error<1e-11
    idx=np.random.default_rng(891).integers(0,64,(4000,64));rows=[];comparisons={}
    for n in names:
        data=[r for r in per_prefix if r['arm']==n];row=dict(arm=n)
        for field in ['greedy_gain_vs_baseline','stochastic_gain_vs_baseline','stochastic_gain_vs_acquired','entropy','kl_vs_acquired','unseen_route_probability']:
            v=np.array([r[field] for r in data]);row[field]=float(v.mean())
            if 'gain' in field:row[field+'_bootstrap95']=np.quantile(v[idx].mean(1),[.025,.975]).tolist()
        rows.append(row)
    for n in ['forward_static','reference_learner']:
        left=[r for r in per_prefix if r['arm']==n];right=[r for r in per_prefix if r['arm']=='prefix']
        for metric in ['greedy_gain_vs_baseline','stochastic_gain_vs_baseline']:
            v=np.array([l[metric]-r[metric] for l,r in zip(left,right)]);comparisons[n+'_over_prefix_'+metric]=dict(gain=float(v.mean()),bootstrap95=np.quantile(v[idx].mean(1),[.025,.975]).tolist())
    checkpoint=out.with_suffix('.pt');torch.save(dict(per_future_losses=torch.stack(truth),probabilities={n:torch.stack(v) for n,v in all_pi.items()}),checkpoint)
    paths=['experiments/credit/diagnose_joint_policy_objective.py','experiments/credit/shared_carrier_paired_replay.py',*old.SOURCES]
    result=dict(status='completed',battle='B1/R1',tag=a.tag,rows=rows,comparisons=comparisons,source_result=SOURCE,source_result_sha256=hashlib.sha256((ROOT/SOURCE).read_bytes()).hexdigest(),source_checkpoint_sha256=source['checkpoint_sha256'],checkpoint=str(checkpoint.relative_to(ROOT)),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),greedy_replay_error=max_replay_error,teacher_worlds=64*8*72,readout_updates=64*8*72*4,supported_flops=work.get_total_flops(),whole_wall_s=time.monotonic()-start,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},scope='Read-only error analysis of one saved training comparison on its same64 DEV prefixes/eight futures. All72 true independently adapting route worlds paid; no fit/tuning/TEST. Exact inference-policy expected utility uses saved0.9softmax+0.1uniform law; greedy utility independently reproduced. Baseline and acquired128 serve separately as references. Saved full per-future losses are DEV evidence, prohibited as future TRAIN labels. Shared source pretraining/acquisition/fit/earlier verification additional. Supported FLOPs omit unsupported/scalar/special/optimizer arithmetic, in wall.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
