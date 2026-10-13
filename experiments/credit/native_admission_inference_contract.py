"""B10 frozen native admission: selected-value execution preserves stochastic numerical state."""
import argparse,copy,hashlib,json,resource,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments/credit'))
from market_native_remote_admission import restored,load,warm,receipt,control,NativeMarketCredit
from experiments.market.paired_clock_credit import pair_window

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args();torch.set_num_threads(1);torch.set_default_dtype(torch.float64);begin=time.monotonic()
 model,inputs=restored();t,m,path=load('2026-08-24',256);inputs[str(path.relative_to(ROOT))]=sha(path);state=warm(model,t,m,128);tokens=[]
 for block in range(8):
  k=127+block*8;old=control.detach_state(state);state,own,projected,scores=pair_window(model,state,t,m,k,8);tokens.append(receipt(model,old,state,own,projected,t,m,k,8,scores))
 torch.manual_seed(19101);critic=NativeMarketCredit(264).double();critic.core.head=torch.nn.Linear(critic.core.total_payload,1).double() # nonzero outputs expose full body parity
 records=[];snapshots=[]
 for mode in ['training_values','selected_values']:
  native=copy.deepcopy(critic);native.train(mode=='training_values');torch.manual_seed(19111);carry=None;values=[];tick=time.monotonic()
  with torch.no_grad():
   for features,events in tokens:out,carry=native(features,events,carry);values.append(out.item())
  records.append(dict(mode=mode,wall_s=time.monotonic()-tick,values=values,selected_updates=native.selected_updates,candidate_scores=native.candidate_scores,counterfactual_values=native.counterfactual_values,rng_sha256=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()))
  snapshots.append(carry)
 errors=[]
 for field in ['memories','arrivals','contexts']:
  x=getattr(snapshots[0].state,field);y=getattr(snapshots[1].state,field);assert x.keys()==y.keys()
  for key in x:
   pairs=zip(x[key],y[key]) if field=='contexts' else [(x[key],y[key])]
   errors.extend(float((u-v).abs().max()) for u,v in pairs)
 error=max(abs(x-y) for x,y in zip(records[0]['values'],records[1]['values']));state_error=max(errors);assert error<1e-12 and state_error<1e-12
 assert records[0]['rng_sha256']==records[1]['rng_sha256'];assert records[0]['selected_updates']==records[1]['selected_updates'];assert records[0]['candidate_scores']==records[1]['candidate_scores'];assert records[1]['counterfactual_values']==0
 sources=['experiments/credit/native_admission_inference_contract.py','experiments/credit/market_native_remote_admission.py','experiments/credit/native_market_credit.py','sleeping_machines/addressed_event_heads.py','sleeping_machines/parallel_head_race_language.py','sleeping_machines/sparse_race_language.py','experiments/market/NATIVE_INFERENCE_ADMISSION_CONTRACT.md']
 r=dict(status='completed',battle='B10',rows=records,output_max_error=error,numerical_state_max_error=state_error,source_sha256={n:sha(ROOT/n) for n in sources},input_sha256=inputs,whole_wall_s=time.monotonic()-begin,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scope='Actual market-derived paired-credit receipts and native admission body with nonzero diagnostic head. Frozen parameters/no-gradient forward; same stochastic draw stream. Selected-only values preserve outputs/memories/arrivals/contexts and RNG, while keys still scored. Eight receipt correctness/operation-count contract; not trained-controller trajectory, whole-market learning latency or hardware energy benchmark.')
 with (ROOT/f'experiments/results/market/{a.tag}.json').open('x') as f:json.dump(r,f,indent=2);f.write('\n')
 print(json.dumps(dict(output_error=error,state_error=state_error,rows=records)),flush=True)
if __name__=='__main__':main()
