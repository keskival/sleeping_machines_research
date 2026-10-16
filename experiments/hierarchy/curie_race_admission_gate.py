"""Source-bound numerical admission for Curie's current CERL/RMR implementation."""
import argparse,hashlib,json,resource,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'experiments'),str(ROOT)]
import torch
p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--model',choices=['cerl','rmr'],required=True);a=p.parse_args()
torch.set_num_threads(1);torch.manual_seed(0);start=time.monotonic()
import r1_cerl,r1_rmr
checks={'base':r1_cerl.contracts()}
if a.model=='rmr':checks['memory']=r1_rmr.contracts()
else:checks['messages']=r1_cerl.contracts(cause_messages=True)
assert all(v['status']=='passed' for v in checks.values()),checks
sources=['experiments/hierarchy/curie_race_admission_gate.py','experiments/r1_cerl.py','experiments/r1_rmr.py','experiments/r1_deep_keyed_lm.py','experiments/r1_token_keyed_lm.py']
r=dict(status='completed',battle='R1',stage='current Curie numerical admission',gate='PASS',checks=checks,elapsed_s=time.monotonic()-start,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,source_sha256={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sources})
out=ROOT/f'experiments/results/diagnostics/{a.tag}.json'
with out.open('x') as f:json.dump(r,f,indent=2);f.write('\n')
print(json.dumps(r),flush=True)
