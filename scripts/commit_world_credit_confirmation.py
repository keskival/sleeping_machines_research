"""Commit only completed Curie stage evidence and matching publication artifacts."""
import argparse
import fcntl
import hashlib
import json
import subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);ap.add_argument('--seed',type=int,choices=[194,195,196],required=True);a=ap.parse_args()
 receipt=json.loads((ROOT/f'.git/credit-report-preview/{a.tag}/publication.json').read_text());assert receipt['status']=='completed'
 paths=list((ROOT/'experiments/results/causal_induction').glob(f'curie_world_credit_confirm_20261012_v1_s{a.seed}_*.json'))
 assert any(p.name.endswith('_decision.json') for p in paths)
 for p in paths:assert json.loads(p.read_text())['status']=='completed'
 with open('/tmp/aws-language-publication.lock','a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  assert subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()=='main'
  # Another publisher may have updated a canonical document since this receipt.
  # Keep its current artifact and commit this stage's results regardless.
  matched=[]
  for name,digest in receipt['artifact_sha256'].items():
   p=ROOT/name
   if hashlib.sha256(p.read_bytes()).hexdigest()==digest:matched.append(p)
  paths+=matched
  for name in ['experiments/HANDOFF.md','experiments/causal_induction/NATIVE_SENSOR.md']:
   p=ROOT/name
   if f'stage{a.seed} complete' in p.read_text():paths.append(p)
  relative=[str(p.relative_to(ROOT)) for p in dict.fromkeys(paths)]
  subprocess.run(['git','add','--',*relative],cwd=ROOT,check=True)
  diff=subprocess.run(['git','diff','--quiet','HEAD','--',*relative],cwd=ROOT)
  if diff.returncode==1:subprocess.run(['git','commit','--only','-m',f'Record completed native world-credit confirmation seed{a.seed}','--',*relative],cwd=ROOT,check=True)
  else:assert diff.returncode==0
if __name__=='__main__':main()
