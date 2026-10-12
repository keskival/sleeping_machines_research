"""Publish completed direct-credit numerical evidence and current report parts."""
import argparse
import fcntl
import hashlib
import json
import subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args()
 path=ROOT/'experiments/results/causal_induction/curie_direct_route_depth_20261012_v1_contract.json';r=json.loads(path.read_text());assert r['status']=='completed'
 for group in ['source_sha256','parent_sha256']:
  for name,h in r[group].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,name
 with open('/tmp/aws-language-publication.lock','a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  p=ROOT/'experiments/HANDOFF.md';s=p.read_text();marker='**Curie direct-credit depth/utility contracts — 12 October,'
  paragraph='**Curie direct-credit depth/utility contracts — 12 October, complete.** B5/R132float64score fixtures2/4/8/32routes pass fixed-belief L1 error bound, producer-covariance and pair-conditioned mean identities. Actual clipped/unclipped Adam proposals match0error; delayed-utility identity8.67e-19 and finite differences2.80e-10. Direct route-score error need not compound local estimator errors through depth; producer geometry, belief accuracy and all response/discovery work remain required. The actual update response projects future cotangents into one scalar teacher per retained credit control. Numerical proof support only: no new learned optimizer, sparse-hardware measurement or neural-depth scaling result. Source-bound result curie_direct_route_depth_20261012_v1_contract; theory DIRECT_ROUTE_CREDIT_DEPTH_AND_DELAYED_UTILITY.md. Native confirmation unchanged: fresh-system194passes, fresh-source195gatepasses/worldlast8NLL0.597551 and four-arm deployments run sequentially;196only after195pass. No variant fit or public scoring admitted.'
  if marker in s:
   start=s.index(marker);end=s.index('\n',start);s=s[:start]+paragraph+s[end:]
  else:s=paragraph+'\n\n'+s
  p.write_text(s)
  p=ROOT/'experiments/causal_induction/NATIVE_SENSOR.md';s=p.read_text();marker='## Unchanged fresh-system confirmation'
  text='The direct route-score error bound and delayed optimizer-utility response are derived in `experiments/theory/DIRECT_ROUTE_CREDIT_DEPTH_AND_DELAYED_UTILITY.md`. Their32score fixtures and actual clipped/unclipped Adam operator checks pass. These are conditional numerical contracts, not trained depth-efficiency evidence.\n\n'
  if 'The direct route-score error bound' not in s:s=s.replace(marker,text+marker,1)
  p.write_text(s)
 subprocess.run([str(ROOT/'.venv-docker/bin/python'),'scripts/build_credit_report_parts.py','--tag',a.tag],cwd=ROOT,check=True)
if __name__=='__main__':main()
