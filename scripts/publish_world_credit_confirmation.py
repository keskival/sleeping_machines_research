"""Publish each completed unchanged native-credit confirmation stage."""
import argparse
import fcntl
import hashlib
import json
import subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);ap.add_argument('--seed',type=int,choices=[194,195,196],required=True);a=ap.parse_args()
 path=ROOT/f'experiments/results/causal_induction/curie_world_credit_confirm_20261012_v1_s{a.seed}_decision.json'
 data=json.loads(path.read_text());assert data['status']=='completed'
 for group in ['source_sha256','parent_sha256']:
  for name,digest in data[group].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
 paragraph='**Curie unchanged world-credit confirmation — 12 October, stage'+str(a.seed)+' complete.** Fixed fresh systems9500/9501,9600/9601,9700/9701,9800/9801,8/16factors and generating-depth4/8,1024prequential targets per arm. '+', '.join(f"{r['arm']} NLL{r['pooled_nll']:.6f}/complete stage{r['complete_acquisition_and_reuse_stage_wall_s']:.3f}s" for r in data['rows'])+'. Four-control gate '+('passes' if data['confirmation_gate'] else 'fails')+'; learned gains '+', '.join(f"{k}:{g:+.6f}" for k,g in data['quality_gains'].items())+'. Next: '+data['next_decision']+'. All source/Adam fingerprints, native outcome/shared-memory learning, decoded alternatives and causal8-feedback schedules match within each seed. Source4096presentations charged once per8deployment cohort,5120full denominator; existing194 source reused,195/196 fresh acquisition only after preceding pass. Three staged passes required; early failure stops later acquisitions. Preserve original completed primary failure and additional persistence-control pass. No new architecture fit admitted, publicTEST, neural-depth scaling or efficiency claim. Protocol causal_induction/WORLD_CREDIT_CONFIRMATION_PROTOCOL.md; tmux curie_world_credit_confirm_20261012; logs.git/curie-world-credit-confirm-preview/20261012_v1; oneCPU/RSS2.2GBtargets/1.5GBsource/9GiBhostfloor.'
 with open('/tmp/aws-language-publication.lock','a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  assert subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()=='main'
  assert not subprocess.check_output(['git','diff','--name-only','--diff-filter=U'],cwd=ROOT,text=True).strip()
  p=ROOT/'experiments/HANDOFF.md';s=p.read_text();marker='**Curie unchanged world-credit confirmation — 12 October,';start=s.index(marker);end=s.index('\n',start);p.write_text(s[:start]+paragraph+s[end:])
  p=ROOT/'experiments/causal_induction/NATIVE_SENSOR.md';s=p.read_text();marker='## Unchanged fresh-system confirmation'
  if marker in s:s=s[:s.index(marker)].rstrip()+'\n'
  p.write_text(s+'\n'+marker+'\n\n'+paragraph+'\n')
 python=str(ROOT/'.venv-docker/bin/python')
 subprocess.run([python,'scripts/update_curie_learning_status.py'],cwd=ROOT,check=True)
 subprocess.run([python,'scripts/build_credit_report_parts.py','--tag',a.tag],cwd=ROOT,check=True)
if __name__=='__main__':main()
