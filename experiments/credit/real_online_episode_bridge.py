"""B10/R1 bounded real-data stateful frozen versus online native BPTT bridge.
This measures adaptation headroom; learned-credit comparison is a later gate.
"""
import argparse,hashlib,json,resource,sys,time
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'experiments'));sys.path.insert(0,str(ROOT/'experiments/market'))
import b10_tpp as market
from episode_state import consume as market_consume,revealed_log_probability,detach_state
from token_episode_state import consume as token_consume,revealed_nll
from r1_token_keyed_lm import KeyedTokenLM,load
SOURCES=['experiments/credit/real_online_episode_bridge.py','experiments/credit/token_episode_state.py','experiments/market/episode_state.py','experiments/market/b10_tpp.py','experiments/r1_token_keyed_lm.py','experiments/tpp/race_tpp_v5.py']

def run(model,observations,domain,adapt,context,group,lr):
    model.eval();model=deepcopy(model);opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=0.) if adapt else None
    state=None;pending=None;losses=[];scores=[];updates=0;norms=[];begin=time.monotonic();traces=[]
    def consume(index,state):
        if domain=='market':return market_consume(model,observations[0][index:index+1],observations[1][index:index+1],state,parameter_version=updates)
        token=observations[index:index+1];return token_consume(model,token,state,parameter_version=updates)
    # Replay observed context with no graph; last context event establishes
    # the issued first scored forecast under the declared credit horizon.
    with torch.no_grad():
        for k in range(context-1):pending,state=consume(k,state)
    with torch.set_grad_enabled(adapt):pending,state=consume(context-1,detach_state(state))
    episode_resets=0;warm={k:v.detach().clone() for k,v in model.state_dict().items()}
    total=len(observations[0]) if domain=='market' else len(observations)
    for k in range(context,total):
        with torch.set_grad_enabled(adapt):
            assert pending['parameter_version']==updates
            loss=(-revealed_log_probability(model,pending,observations[0][k:k+1],observations[1][k:k+1]) if domain=='market' else revealed_nll(pending,observations[k:k+1])).mean()
            assert torch.isfinite(loss);scores.append(float(loss.detach()));issued_version=pending['parameter_version']
            if adapt:losses.append(loss)
            eos=domain=='language' and int(observations[k])==50256
            if adapt and (len(losses)==group or eos):
                opt.zero_grad(set_to_none=True);torch.stack(losses).mean().backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();updates+=1;norms.append(float(norm));losses=[];state=detach_state(state)
            if eos:
                model.load_state_dict(warm);state=None;episode_resets+=1
                if adapt:opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=0.)
            traces.append(dict(observation=k,issued_parameter_version=issued_version,next_parameter_version=updates,episode_reset=eos))
            # No new forecast is needed after the last scored observation.
            if k+1<total:pending,state=consume(k,state)
    # Unused final partial gradient accumulation is not applied; all work spent
    # producing its forecasts is still included in wall/work accounting.
    bins=[float(np.mean(scores[i:i+64])) for i in range(0,len(scores),64)]
    return dict(arm='online_native_bptt' if adapt else 'stateful_frozen',nll=float(np.mean(scores)),targets=len(scores),updates=updates,unapplied_tail_losses=len(losses),episode_resets=episode_resets,wall_s=time.monotonic()-begin,block_nll=bins,forecast_versions=traces,gradient_norm_max=max(norms,default=0.),optimizer_state_elements=0 if opt is None else sum(v.numel() for s in opt.state.values() for v in s.values() if isinstance(v,torch.Tensor)))

def measure(a):
    results={};warm_start={};start=time.monotonic();raw=np.load(market.NPZ/f'{market.TRAIN[0]}.npz');n=a.targets+a.context
    times=torch.tensor((raw['t'][:n]-raw['t'][0])*1e-6,dtype=torch.float64);marks=torch.tensor(raw['mark'][:n].astype(np.int64));gaps=torch.diff(times[:a.context]);positive=gaps[gaps>0].numpy();assert len(positive)
    torch.manual_seed(181);model=market.Race(positive,16,4).m.double();observations=(times,marks)
    warm_start['market']=dict(kind='same initialization using prior context-only gap statistics; no pretrained weights',context_events=a.context,scope='First-pass causal headroom probe, not the completed B10 fit or its headline quality.')
    results['market']=[run(model,observations,'market',adapt,a.context,a.group,a.lr) for adapt in (False,True)]
    record_path=ROOT/'experiments/results/token_language/curie_r1g2_keyed1_4M_s1_20261008T1840Z.json';record=json.loads(record_path.read_text());p=record['args'];checkpoint=ROOT/record['checkpoint'];model=KeyedTokenLM(p['d'],p['modes'],p['layers'],p['heads'],p['dk'],p['keyed'],p['dropout']);model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True));model.double();data=load(ROOT/'data/fineweb_gpt2/fineweb_train_000001.bin');ids=torch.tensor(np.array(data[p['train_tokens']:p['train_tokens']+n],dtype=np.int64))
    warm_start['language']=dict(checkpoint=str(checkpoint.relative_to(ROOT)),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),pretrained_tokens=p['train_tokens'],prior_fit_wall_s=record['wall_s'],fresh_suffix_offset=p['train_tokens'],context_tokens=a.context,scope='Available trained keyed-token member, normalized GPT-2 vocabulary; historical selection/public score not repeated.')
    results['language']=[run(model,ids,'language',adapt,a.context,a.group,a.lr) for adapt in (False,True)]
    for domain,rows in results.items():assert rows[0]['targets']==rows[1]['targets']==a.targets
    return dict(domains=results,warm_start=warm_start,whole_wall_s=time.monotonic()-start,scope='One short contiguous real TRAIN episode/fragment per domain; score issued forecasts before feedback updates. Stateful frozen versus native online BPTT only; equal context/targets, float64, eval-mode dropout, group-sized truncated credit with numerical carry. Language resets fast weights/optimizer/forward state after observed EOS. No learned-credit efficiency or dataset benchmark claim. Complete model, readout and optimizer work included in wall time; no FLOP/energy inference from wall.')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);ap.add_argument('--mode',choices=['smoke','measure'],required=True);ap.add_argument('--targets',type=int,default=1024);ap.add_argument('--context',type=int,default=128);ap.add_argument('--group',type=int,default=32);ap.add_argument('--lr',type=float,default=.0003);a=ap.parse_args()
    if a.mode=='smoke':a.targets=32;a.context=32;a.group=8
    output=ROOT/'experiments/results/credit'/f'{a.tag}.json';assert not output.exists();torch.set_num_threads(1);metrics=measure(a);result=dict(status='completed',tag=a.tag,battle='B10/R1',args=vars(a),metrics=metrics,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,source_sha256={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in SOURCES});output.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
