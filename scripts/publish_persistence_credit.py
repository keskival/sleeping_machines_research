"""Publish the completed additional strong-credit control, keeping the primary gate."""
import argparse
import fcntl
import hashlib
import json
import subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args()
 data=json.loads((ROOT/'experiments/results/causal_induction/curie_persistence_credit_20261012_v1_decision.json').read_text())
 assert data['status']=='completed'
 for group in ['source_sha256','parent_sha256']:
  for name,digest in data[group].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
 outcome='passes' if data['strong_control_gate'] else 'fails'
 next_step=('unchanged new-actor/new-system confirmation against persistence credit' if data['strong_control_gate'] else 'native response-conditioned outcome credit against issued forward-belief teaching')
 paragraph='**Curie strong conditional-credit control — 12 October, complete.** B5/R1 same common native world/actor/Adam fingerprint, eight independent128-target deployments, same seeds/data/actual-outcome teaching and independently learned shared memory. Persistence posterior uses only public last-observed target bit and source counts, updated after delivery; issued probabilities match the independent causal reference exactly. '+f"Persistence credit NLL{data['persistence_credit_nll']:.6f}; learned NLL{data['learned_credit_nll']:.6f}; learned gain{data['quality_gains_vs_persistence']['learned']:+.6f}; generating-depth gains "+', '.join(f"{d}:{g:+.6f}" for d,g in data['learned_depth_gains_vs_persistence'].items())+f". Additional strong-control gate {outcome}; original v2 primary gate stays failed. Complete acquisition/reuse stage{data['complete_acquisition_and_reuse_stage_wall_s']:.3f}s,4096source plus1024target presentations; all native critic/alternative/optimizer work retained. Next: "+next_step+'. Protocol causal_induction/PERSISTENCE_CREDIT_PROTOCOL.md; source-bound completed results preserved. V1 smoke inherited an inaccurate predecessor scope string; v2 corrects metadata without changing learning. OneCPU/RSS2.2GB/9GiBhostfloor/cgroup2GiB reserve. No minimal-cost reference, publicTEST, neural-depth scaling or family-level verdict. AWS history-conditioned child stays separate. Logs.git/curie-persistence-credit-preview/20261012_v1.'
 with open('/tmp/aws-language-publication.lock','a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  assert subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()=='main'
  assert not subprocess.check_output(['git','diff','--name-only','--diff-filter=U'],cwd=ROOT,text=True).strip()
  p=ROOT/'experiments/HANDOFF.md';s=p.read_text();marker='**Curie strong conditional-credit control — 12 October,';start=s.index(marker);end=s.index('\n',start);p.write_text(s[:start]+paragraph+s[end:])
  p=ROOT/'experiments/causal_induction/NATIVE_SENSOR.md';s=p.read_text();marker='## Strong conditional-credit diagnostic';
  if marker in s:s=s[:s.index(marker)].rstrip()+'\n'
  s+='\n'+marker+'\n\n'+paragraph.replace('**Curie strong conditional-credit control — 12 October, complete.** ','')+'\n';p.write_text(s)
 python=str(ROOT/'.venv-docker/bin/python')
 subprocess.run([python,'scripts/update_curie_learning_status.py'],cwd=ROOT,check=True)
 subprocess.run([python,'scripts/build_credit_report_parts.py','--tag',a.tag],cwd=ROOT,check=True)
if __name__=='__main__':main()
