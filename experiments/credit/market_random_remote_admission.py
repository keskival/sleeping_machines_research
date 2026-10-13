"""B10 diagnostic: activity-matched random remote credit versus learned frozen admission."""
import argparse,copy,hashlib,json,resource,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments/credit'))
from market_streaming_clock_updates import restored,load,warm,stream,control
from market_native_remote_admission import step
from experiments.market.paired_clock_credit import pair_window

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args();torch.set_num_threads(1);torch.set_default_dtype(torch.float64);begin=time.monotonic()
 parent=ROOT/'experiments/results/market/curie_b10_native_admission_longer_20261013_v2_fit.json';record=json.loads(parent.read_text());frozen=next(x for x in record['rows'] if x['arm']=='frozen');count=int(sum(frozen['admission']));assert count==57
 model,inputs=restored();inputs[str(parent.relative_to(ROOT))]=sha(parent);group=32;context=128;blocks=512
 t,m,path=load('2026-08-24',context+2048);inputs[str(path.relative_to(ROOT))]=sha(path);opt=torch.optim.Adam(model.parameters(),lr=.0003)
 bundle,source=stream((model,opt,warm(model,t,m,context)),t,m,context-1,64,group,'bptt')
 t,m,path=load('2026-08-25',context+blocks*group);inputs[str(path.relative_to(ROOT))]=sha(path);bundle=(*bundle[:2],warm(bundle[0],t,m,context));rows=[]
 for seed in [19105,19106,19107]:
  actor,optimizer,state=copy.deepcopy(bundle);chosen=set(np.random.default_rng(seed).choice(blocks,count,replace=False).tolist());scores=[];tick=time.monotonic()
  for block in range(blocks):
   state,own,projected,losses=pair_window(actor,state,t,m,context-1+block*group,group);scores.append(float(np.mean(losses)));step(actor,optimizer,projected if block in chosen else own)
  rows.append(dict(seed=seed,nll=float(np.mean(scores)),targets=blocks*group,admission_count=len(chosen),admission_indices=sorted(chosen),block_nll=scores,wall_s=time.monotonic()-tick))
 values=[r['nll'] for r in rows];sources=['experiments/credit/market_random_remote_admission.py','experiments/credit/market_streaming_clock_updates.py','experiments/credit/market_native_remote_admission.py','experiments/market/paired_clock_credit.py','experiments/market/projected_clock_state.py','experiments/market/local_eligibility_state.py','experiments/credit/local_clock_eligibility.py','experiments/market/RANDOM_REMOTE_ADMISSION_PROTOCOL.md']
 result=dict(status='completed',battle='B10',rows=rows,acquisition=source,frozen_native_nll=frozen['nll'],random_mean_nll=float(np.mean(values)),random_sample_sd=float(np.std(values,ddof=1)),random_minus_frozen_native=[v-frozen['nll'] for v in values],native_beats_all_random=all(v>frozen['nll'] for v in values),input_sha256=inputs,source_sha256={p:sha(ROOT/p) for p in sources},whole_wall_s=time.monotonic()-begin,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scope='Three predeclared random schedule seeds, same restored actor/exact source Adam acquisition/DEV data/group32 and57of512remote updates as completed frozen native. Activity count calibrated from reusedDEV native result; this is a development attribution diagnostic, not untouched evaluation or proof of learned inductive biases. Uniform random schedules fixed before each replay, no future-outcome use in updates. Both endpoints computed; no activity savings claim. Common original5750.687s predictor fitting additionally charged. No controller in random arms; native source/teacher/inference cost remains paid in its existing record.')
 with (ROOT/f'experiments/results/market/{a.tag}.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps({k:result[k] for k in ['frozen_native_nll','random_mean_nll','random_sample_sd','random_minus_frozen_native','native_beats_all_random']}),flush=True)
if __name__=='__main__':main()
