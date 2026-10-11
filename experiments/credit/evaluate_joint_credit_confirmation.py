"""B1/R1 fixed fresh full-support three-continuation-stream confirmation."""
import argparse,hashlib,json,resource,time
from copy import deepcopy
import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode
import sparse_paired_actor_critic as old
from shared_carrier_paired_replay import replay_pairs
ROOT=old.ROOT
TAG='curie_joint_forward_credit_confirmation_20261011_v1'
ARMS=['prefix','forward_static','reference_learner']

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args()
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64);start=time.monotonic()
    out=ROOT/f'experiments/results/credit/{a.tag}.json';assert not out.exists()
    paths=['experiments/results/credit/curie_joint_forward_credit_20261011_v1_fit.json',*[f'experiments/results/credit/{TAG}_r{r}.json' for r in [1,2]]]
    results=[];models={};carriersha=None;carrier,baseline,tau,_,_=old.initial()
    prior,acq=old.load(f'experiments/results/credit/{old.TAG}_acquire.json')
    initial=deepcopy(baseline);initial.load_state_dict(acq['states']['full_fixed']['actor'])
    models['acquired128']=initial;models['baseline']=deepcopy(baseline)
    for r,path in enumerate(paths):
        result,cp=old.load(path);results.append(result)
        for p,digest in result['source_sha256'].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==digest,p
        assert result['source_checkpoint_sha256']==prior['checkpoint_sha256']
        for arm in ARMS:
            state=cp['states'][arm];assert state['cursor']==512 and len(state['history'])==512
            net=deepcopy(baseline);net.load_state_dict(state['actor']);models[f'r{r}_{arm}']=net
    per_prefix=[];truth=[];all_pi={n:[] for n in models};routes=torch.arange(8).repeat_interleave(9).tolist();actions=old.native.base.actions().repeat(8,1)
    with FlopCounterMode(display=False) as work:
        for i in range(64):
            t,m=old.native.base.batch(148124000+i,size=1,pairs=2)
            with torch.no_grad():
                x=old.native.prefix(carrier,t[:,:4],m[:,:4],3,list(range(8)))
                scores={n:old.scores(net,x,512).flatten() for n,net in models.items()}
                pi={n:old.probabilities(v.reshape(1,8,9),tau).flatten() for n,v in scores.items()}
                selected={n:int(v.argmin()) for n,v in scores.items()}
            rng=np.random.default_rng(89000000+i);values=[]
            for _ in range(8):
                ft,fm=old.native.prior.continuation(t,m,2,rng)
                losses,_=replay_pairs(carrier,ft,fm,routes,actions,.05);values.append(losses.mean(1))
            q=torch.stack(values);truth.append(q);q=q.mean(0)
            for n in models:all_pi[n].append(pi[n])
            for r in range(3):
                for arm in ARMS:
                    name=f'r{r}_{arm}'
                    per_prefix.append(dict(replicate=r,prefix=i,arm=arm,stochastic_gain_vs_acquired=float(((pi['acquired128']-pi[name])*q).sum()),stochastic_gain_vs_baseline=float(((pi['baseline']-pi[name])*q).sum()),greedy_gain_vs_baseline=float(q[selected['baseline']]-q[selected[name]])))
            if (i+1)%16==0:print(json.dumps(dict(prefixes=i+1,wall_s=time.monotonic()-start)),flush=True)
    arrays={arm:{metric:np.array([[next(v[metric] for v in per_prefix if v['replicate']==r and v['prefix']==i and v['arm']==arm) for i in range(64)] for r in range(3)]) for metric in ['stochastic_gain_vs_acquired','stochastic_gain_vs_baseline','greedy_gain_vs_baseline']} for arm in ARMS}
    rng=np.random.default_rng(901);seeds=rng.integers(0,3,(4000,3));prefixes=rng.integers(0,64,(4000,64))
    def ci(value):return np.quantile(value[seeds[:,:,None],prefixes[:,None,:]].mean((1,2)),[.025,.975]).tolist()
    rows=[];comparisons={}
    for arm in ARMS:
        for r in range(3):
            cost=next(v for v in results[r]['cumulative_fits'] if v['arm']==arm)
            rows.append(dict(replicate=r,arm=arm,**{k:float(v[r].mean()) for k,v in arrays[arm].items()},whole_fit_s=cost['wall_s'],whole_fit_supported_gflops=cost['supported_flops']/1e9,fit_supported_mflops_per_target=cost['supported_flops']/cost['factual_targets']/1e6))
    for first,second in [('reference_learner','prefix'),('forward_static','prefix'),('reference_learner','forward_static')]:
        for metric in ['stochastic_gain_vs_baseline','greedy_gain_vs_baseline']:
            value=arrays[first][metric]-arrays[second][metric]
            comparisons[f'{first}_over_{second}_{metric}']=dict(gain=float(value.mean()),per_stream_gain=value.mean(1).tolist(),hierarchical_bootstrap95=ci(value))
    primary=comparisons['reference_learner_over_prefix_stochastic_gain_vs_baseline']
    improvement=arrays['reference_learner']['stochastic_gain_vs_acquired']
    gate=primary['hierarchical_bootstrap95'][0]>0 and min(primary['per_stream_gain'])>0 and improvement.mean()>0
    checkpoint=out.with_suffix('.pt');torch.save(dict(per_future_losses=torch.stack(truth),probabilities={n:torch.stack(v) for n,v in all_pi.items()}),checkpoint)
    sources=['experiments/credit/evaluate_joint_credit_confirmation.py','experiments/credit/JOINT_FORWARD_CREDIT_CONFIRMATION.md','experiments/credit/shared_carrier_paired_replay.py',*old.SOURCES]
    result=dict(status='completed',battle='B1/R1',tag=a.tag,confirmation_gate=bool(gate),rows=rows,comparisons=comparisons,per_prefix=per_prefix,reference_gain_vs_acquired=float(improvement.mean()),reference_gain_vs_acquired_hierarchical95=ci(improvement),teacher_worlds=64*8*72,readout_updates=64*8*72*4,supported_flops=work.get_total_flops(),whole_wall_s=time.monotonic()-start,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,source_results={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},checkpoint=str(checkpoint.relative_to(ROOT)),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},scope='Prospective fresh64-prefix/eight-future DEV, fixed registered scoring/endpoint/settings. Three continuation streams share actor seed170 and valid128 acquisition; r0 reused, r1/2 new data/draws/futures. Not three native model initializations. Exact full72-route independent learner scores, paired stream/context bootstrap. All teacher worlds/readout updates charged; per-arm whole fit includes inherited acquisition. Prior diagnostics and shared verification additional. Supported FLOPs omit unsupported/scalar/special/optimizer arithmetic, included in wall. DEV labels retained as evidence, prohibited as TRAIN. One synthetic development confirmation, no TEST/public benchmark/depth/hardware claim.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
