"""Validate RMR-2 numerics under the guarded R1 pipeline, without fitting."""
import argparse,hashlib,json,resource,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'experiments'),str(ROOT)]
import torch
p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args();torch.set_num_threads(1)
import r1_rmr2
start=time.monotonic();checks=r1_rmr2.contracts();assert checks['status']=='passed'
sources=['experiments/hierarchy/rmr2_admission_gate.py','experiments/r1_rmr2.py','experiments/r1_rmr.py','experiments/r1_cerl.py','experiments/r1_deep_keyed_lm.py','experiments/r1_token_keyed_lm.py']
r=dict(status='completed',battle='R1',stage='RMR2 numerical admission',gate='PASS',checks=checks,elapsed_s=time.monotonic()-start,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,source_sha256={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sources})
with (ROOT/f'experiments/results/diagnostics/{a.tag}.json').open('x') as f:json.dump(r,f,indent=2);f.write('\n')
print(json.dumps(r),flush=True)
