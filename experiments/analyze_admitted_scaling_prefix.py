"""R1 recorded acquisition-prefix integrity for the retained admitted-teacher member."""
import argparse,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def canon(v):
 if isinstance(v,dict):return {k:canon(x) for k,x in v.items() if k not in ('wall_s','learner_fingerprint','memory_effective_rank')}
 if isinstance(v,list):return [canon(x) for x in v]
 return v

def main():
 p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--seed',required=True,type=int);a=p.parse_args();sources={};inputs={};records=[]
 names=([f'aws_bernoulli_teacher_2m_20261013_s6_v2',f'aws_bernoulli_teacher_8m_20261013_s6_v2'] if a.seed==6 else [f'aws_bernoulli_teacher_512k_20261013_s{a.seed}_v2',f'aws_bernoulli_teacher_8m_20261013_s{a.seed}_v3'])
 for tag in names:
  path=ROOT/f'experiments/results/token_language/{tag}.json';r=json.loads(path.read_text());assert r['status']=='completed';inputs[str(path.relative_to(ROOT))]=sha(path);sources.update(r['source_sha256']);records.append(r)
 for name,digest in sources.items():assert sha(ROOT/name)==digest,name
 before,after=records;limit=before['args']['steps']
 assert {k:v for k,v in before['args'].items() if k not in ('tag','steps')}=={k:v for k,v in after['args'].items() if k not in ('tag','steps')}
 exact=canon(before['curve'])==canon([x for x in after['curve'] if x['step']<=limit])
 sources[str(Path(__file__).relative_to(ROOT))]=sha(Path(__file__))
 d=dict(status='completed',battle='R1',gate='PASS' if exact else 'STOP',seed=a.seed,recorded_prefix_exact=exact,prefix_steps=limit,prefix_targets=before['presentations_total'],source_sha256=sources,input_sha256=inputs,scope='Same admitted-teacher member/RNG/data/settings; only total acquisition size differs. Exact recorded loss/teacher/gradient/routing prefix required, excluding elapsed wall, budget-dependent fingerprint placement and nondifferentiable SVD rounding. This is a recorded-prefix witness, not equality proof of every unrecorded learner state.')
 with (ROOT/f'experiments/results/diagnostics/{a.tag}.json').open('x') as f:json.dump(d,f,indent=2);f.write('\n')
 print(json.dumps({k:v for k,v in d.items() if k not in ('source_sha256','input_sha256')}),flush=True)
if __name__=='__main__':main()
