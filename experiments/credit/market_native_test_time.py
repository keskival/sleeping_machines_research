"""B10 acquired native market predictor with event-native online credit teaching."""
import argparse,copy,hashlib,json,resource,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments/credit'))
import market_prospective_bias_credit as engine
from native_market_credit import NativeMarketCredit
from real_boundary_stable import control
market=control.market
REFERENCE='curie_b10_race_dev_s0_20261009T1800Z'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def restored():
 path=ROOT/f'experiments/results/market/{REFERENCE}.json';record=json.loads(path.read_text());assert record['status']=='completed' and not record['args']['score_test']
 assert sha(ROOT/'experiments/market/b10_tpp.py')==record['source_sha256']
 tr=[w for day in market.TRAIN for w in market.windows(day,record['args']['train_every'])]
 gaps=np.concatenate([np.diff(w[0]) for w in tr]);pos=gaps[gaps>0];del tr,gaps
 model=market.Race(pos,record['args']['d'],record['args']['n_ln']).double();checkpoint=ROOT/f'experiments/results/market/{REFERENCE}.pt';model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True));model.m.eval()
 return model.m,{str(path.relative_to(ROOT)):sha(path),str(checkpoint.relative_to(ROOT)):sha(checkpoint)}

def load(day,n):
 path=market.NPZ/f'{day}.npz'
 with np.load(path,allow_pickle=False) as raw:
  t=torch.tensor((raw['t'][:n]-raw['t'][0])*1e-6,dtype=torch.float64);m=torch.tensor(raw['mark'][:n].astype(np.int64))
 assert len(t)==n and bool((torch.diff(t)>=0).all());return t,m,path

def warm(model,t,m,context):
 state=None
 with torch.no_grad():
  for k in range(context-1):_,state=control.market_consume(model,t[k:k+1],m[k:k+1],state)
 return control.detach_state(state)

def frozen_stream(model,t,m,state,start,blocks,group):
 scores=[];begin=time.monotonic()
 with torch.no_grad():
  for k in range(start,start+blocks*group):
   forecast,state=control.market_consume(model,t[k:k+1],m[k:k+1],state)
   loss=-control.revealed_log_probability(model,forecast,t[k+1:k+2],m[k+1:k+2]).mean();assert torch.isfinite(loss);scores.append(float(loss))
 return dict(arm='fully_frozen',nll=float(np.mean(scores)),targets=len(scores),wall_s=time.monotonic()-begin,block_nll=[float(np.mean(scores[k:k+group])) for k in range(0,len(scores),group)],audits=0,future_teachers=0)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);ap.add_argument('--mode',choices=['smoke','fit'],required=True);a=ap.parse_args();torch.set_num_threads(1);torch.set_default_dtype(torch.float64);begin=time.monotonic();out=ROOT/f'experiments/results/market/{a.tag}.json';assert not out.exists()
 model,inputs=restored();group=8 if a.mode=='smoke' else 32;context=128;acq=2 if a.mode=='smoke' else 64;reuse=2 if a.mode=='smoke' else 128
 t,m,path=load('2026-08-24',context+acq*group);inputs[str(path.relative_to(ROOT))]=sha(path)
 width=sum(v.numel() for v in control.tensors(control.market_consume(model,t[:1],m[:1])[1]));torch.manual_seed(190+10000);critic=NativeMarketCredit(3*width).double().train()
 opt=torch.optim.Adam(model.parameters(),lr=.0003);cop=torch.optim.Adam(critic.parameters(),lr=.001)
 bundle=(model,critic,opt,cop,warm(model,t,m,context),torch.zeros(1,64,dtype=torch.float64))
 bundle,acquisition=engine.stream(bundle,t,m,context-1,acq,group,'online',190+20000)
 t,m,path=load('2026-08-25',context+reuse*group);inputs[str(path.relative_to(ROOT))]=sha(path)
 bundle=(*bundle[:4],warm(bundle[0],t,m,context),bundle[5]);rows=[]
 rows.append(frozen_stream(copy.deepcopy(bundle[0]),t,m,copy.deepcopy(bundle[4]),context-1,reuse,group))
 for arm in ['bptt','zero','frozen','online']:
  torch.manual_seed(190+30000)
  _,row=engine.stream(bundle,t,m,context-1,reuse,group,arm,190+30000);rows.append(row)
 assert rows[-1]['audits']==rows[-2]['audits']==rows[-3]['audits']
 gains={row['arm']:row['nll']-rows[-1]['nll'] for row in rows[:-1]}
 sources=['experiments/credit/market_native_test_time.py','experiments/credit/native_market_credit.py','experiments/credit/market_prospective_bias_credit.py','experiments/credit/real_feedback_credit_metric_v2.py','experiments/credit/real_boundary_stable.py','experiments/credit/real_learned_boundary_v2.py','experiments/credit/real_online_episode_bridge.py','experiments/market/episode_state.py','experiments/market/b10_tpp.py','experiments/tpp/race_tpp_v5.py','sleeping_machines/addressed_event_heads.py','sleeping_machines/parallel_head_race_language.py','sleeping_machines/sparse_race_language.py','sleeping_machines/route_activation_memory.py','experiments/market/NATIVE_TEST_TIME_PROTOCOL.md']
 result=dict(status='completed',battle='B10',tag=a.tag,args=vars(a),acquisition=acquisition,rows=rows,online_gain=gains,quality_gate=all(v>0 for v in gains.values()),source_sha256={p:sha(ROOT/p) for p in sources},input_sha256=inputs,whole_wall_s=time.monotonic()-begin,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,critic_parameters=sum(p.numel() for p in critic.parameters()),actor_parameters=sum(p.numel() for p in model.parameters()),scope='Existing full18,164parameter market predictor/checkpoint retained; new native hard-race critic, causal per-mark activation memory and forward-response-conditioned actual future-Adam teaching. Common acquired actor/critic/optimizer state in all arms; Aug24 TRAIN acquisition, Aug25 DEV prequential adaptation with score before update. Full six-mark plus1us recording-cell timing objective. Dense temporal actor retained; native critic does not implement shared actor/critic bank reads or outcome-law posterior projection. No sealed TEST, cross-day/full-month, trading-profit, sparse hardware or benchmark win. Frozen critic still evolves numerical memory; fully_frozen freezes actor parameters too. All teacher/discovery/alternative/optimizer work included in measured execution.')
 with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(dict(tag=a.tag,online_gain=gains,wall_s=result['whole_wall_s'])),flush=True)
if __name__=='__main__':main()
