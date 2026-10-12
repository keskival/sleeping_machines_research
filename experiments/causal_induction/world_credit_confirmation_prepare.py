"""Fixed fresh public systems for unchanged native-world confirmation."""
import json
from pathlib import Path
import numpy as np
from nonlinear_generate import episode,contracts
from world_credit_pipeline import ROOT,RESULTS,sha
TAG='curie_world_credit_confirm_20261012_v1_prepare'
def main():
 out=RESULTS/(TAG+'.json');assert not out.exists()
 folder=ROOT/'data/causal_induction'/TAG;folder.mkdir(parents=True,exist_ok=False)
 rows=[]
 for group,(depth,width,seed) in enumerate([(4,8,9500),(4,16,9600),(8,8,9700),(8,16,9800)]):
  for i,variant in enumerate(['base','rewire']):
   public,_=episode(seed+i,depth,width,pulses=128,switch=variant=='rewire')
   path=folder/f'case{2*group+i}_{seed+i}_{variant}.npz';np.savez(path,**public)
   rows.append(dict(case=2*group+i,seed=seed+i,depth=depth,width=width,variant=variant,path=str(path.relative_to(ROOT)),sha256=sha(path),targets=128,events=len(public['events'])))
 sources=['experiments/causal_induction/nonlinear_generate.py','experiments/causal_induction/world_credit_confirmation_prepare.py','experiments/causal_induction/WORLD_CREDIT_CONFIRMATION_PROTOCOL.md']
 result=dict(status='completed',battle='B5/R1',tag=TAG,contracts=contracts(),records=rows,source_sha256={n:sha(ROOT/n) for n in sources},scope='Fixed fresh public systems only; no model fit or score. Private generating structure discarded and never supplied to learners.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(tag=TAG,systems=len(rows))),flush=True)
if __name__=='__main__':main()
