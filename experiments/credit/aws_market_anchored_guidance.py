"""B10 acquired AWS actor plus demonstrated native future-taught credit guidance.

Reuses the source-bound native early-request learner; no Curie checkpoint.
Actual actor/Adam source acquisition and chronological stream match the AWS
online reference. Numerical state persists, including the calendar-day gap.
"""
import argparse,copy,json,resource,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments/market'));sys.path.insert(0,str(ROOT/'experiments/credit'))
import market_native_early_request as guide
import market_streaming_clock_updates as exact
from experiments.market.episode_state import initial,detach_state
from b10_matched_data import prepare,digest
from b10_matched_models import MatchedRace
from b10_online_stream import forecast,revealed_terms,detach
from experiments.market.owned_clock_dispatch import consumer
guide.control.market_consume=consumer()
from experiments.market.anchored_native_guidance import anchored_stream
guide.stream=anchored_stream(guide.stream)

def load(day,n,origin):
 p=ROOT/f'data/binance/npz/{day}.npz';z=np.load(p);return torch.tensor((z['t'][:n]-origin)*1e-6),torch.tensor(z['mark'][:n].astype(np.int64)),p

def main():
 p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--mode',choices=['smoke','fit'],required=True);a=p.parse_args();torch.set_num_threads(1);torch.set_default_dtype(torch.float64);began=time.monotonic();data,receipt=prepare();gaps=np.concatenate([np.diff(t) for t,m in data['train']]);gaps=gaps[gaps>0];model=MatchedRace(gaps,'native');base=ROOT/'experiments/results/market/aws_b10_stable_native_s0_20261014_v1.json';r=json.loads(base.read_text());assert r['status']=='completed';weights=ROOT/r['final_weights'];model.load_state_dict(torch.load(weights,weights_only=False)['model']);model.eval();actor=model.m
 group=32;acquisition=2 if a.mode=='smoke' else 64;blocks=2 if a.mode=='smoke' else 512
 torch.manual_seed(19101);critic=guide.NativeMarketCredit(264).double();critic.core.head=torch.nn.Linear(critic.core.total_payload,1).double();torch.nn.init.zeros_(critic.core.head.weight);torch.nn.init.zeros_(critic.core.head.bias)
 opt=torch.optim.AdamW(actor.parameters(),lr=.0001,weight_decay=1e-4,foreach=False);cop=torch.optim.Adam(critic.parameters(),lr=.001)
 z=np.load(ROOT/'data/binance/npz/2026-08-24.npz');origin=int(z['t'][0]);t,m,path=load('2026-08-24',acquisition*group+1,origin);state=initial(actor,1);state['last_time']=t[:1].clone();bundle=(actor,opt,state,critic,cop,.001)
 # The source learner uses exact full actor credit during preparation, while
 # separately training native guidance from actual future-update outcomes.
 bundle,source=guide.stream(bundle,t,m,0,acquisition,group,'online',True)
 # Require its prepared actor to match the ordinary online acquisition. This
 # prevents a better/different warmup being hidden as a credit-policy gain.
 reference=copy.deepcopy(model);reference.m=copy.deepcopy(model.m);ropt=torch.optim.AdamW(reference.parameters(),lr=.0001,weight_decay=1e-4,foreach=False);rs=None
 for k in range(0,acquisition*group,group):
  ropt.zero_grad();parameters,rs=forecast(reference,t[k:k+group][None],m[k:k+group][None],rs);loss=-revealed_terms(reference,parameters,t[k:k+group][None],t[k+1:k+group+1][None],m[k+1:k+group+1][None]).mean();loss.backward();torch.nn.utils.clip_grad_norm_(reference.parameters(),1.,foreach=False);ropt.step();rs=detach(rs)
 error=max(float((p-q).detach().abs().max()) for p,q in zip(bundle[0].parameters(),reference.m.parameters()));assert error<1e-8,('Source acquired actor mismatch',error)
 t,m,target=load('2026-08-25',128+blocks*group,origin)
 with torch.no_grad():
  st=bundle[2]
  for k in range(127):_,st=guide.control.market_consume(bundle[0],t[k:k+1],m[k:k+1],st)
 bundle=(bundle[0],bundle[1],detach_state(st),*bundle[3:]);rows=[]
 _,ordinary=exact.stream(bundle[:3],t,m,127,blocks,group,'bptt');rows.append(ordinary)
 for arm in ('frozen','online'):
  torch.manual_seed(19111);final,row=guide.stream(bundle,t,m,127,blocks,group,arm);row['learning_policy']='previously trained native guidance' if arm=='frozen' else 'native guidance also learns online';rows.append(row)
  torch.save(dict(actor=final[0].state_dict(),optimizer=final[1].state_dict(),state=final[2],critic=final[3].state_dict(),critic_optimizer=final[4].state_dict()),ROOT/f'experiments/results/market/{a.tag}_{arm}.pt')
 primary=ROOT/'experiments/results/market/aws_b10_online_native_s0_20261014_v1.json';parity=None
 if a.mode=='fit':
  previous=json.loads(primary.read_text());assert previous['status']=='completed';parity=abs(rows[0]['nll']-previous['rows'][0]['nll']);assert parity<1e-8,('Online BPTT stream mismatch',parity)
 names=['experiments/credit/aws_market_anchored_guidance.py','experiments/credit/market_native_early_request.py','experiments/credit/market_native_remote_admission_selected.py','experiments/credit/native_market_credit.py','experiments/credit/market_streaming_clock_updates.py','experiments/market/paired_clock_credit.py','experiments/market/projected_clock_state.py','experiments/market/local_eligibility_state.py','experiments/market/episode_state.py','experiments/market/b10_stable_clock_head.py','experiments/market/b10_online_stream.py','experiments/market/owned_clock_dispatch.py','experiments/market/anchored_native_guidance.py']
 result=dict(status='completed',battle='B10',mode=a.mode,source_actor_parity_error=error,primary_online_bptt_parity_error=parity,acquisition=source,rows=rows,source_sha256={n:digest(ROOT/n) for n in names},input_sha256={str(p.relative_to(ROOT)):digest(p) for p in [base,weights,path,target]},inherited_fitting_arithmetic=r['fitting_arithmetic'],wall_s=time.monotonic()-began,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scope='Same acquired AWS native actor, AdamW warmup and chronological targets as online references; native persistent state survives gradient boundaries/day gap. Retain exact factual32-event BP and add optional projected-minus-own proposal chosen by native hard-race critic trained on actual future-Adam utility; mean-changing update guidance, not an unbiased gradient replacement. Frozen critic weights still evolve event memory and native actor adapts online. Source guidance preparation plus duplicate exact-acquisition/parity work charged; elapsed execution reported, full arithmetic accounting still needed for efficiency claim. Reused DEV,seed0,Septembersealed; development candidate, not complete family verdict.')
 out=ROOT/f'experiments/results/market/{a.tag}.json'
 with out.open('x') as f:json.dump(result,f,indent=2)
 print(json.dumps(dict(mode=a.mode,source_actor_parity_error=error,nll={x['arm']:x['nll'] for x in rows})),flush=True)
if __name__=='__main__':main()
