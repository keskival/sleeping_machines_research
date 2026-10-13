"""R1 residual-credit integration: off-mode parity or useful-quality promotion."""
import argparse,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def canon(v):
 if isinstance(v,dict):return {k:canon(x) for k,x in v.items() if k not in ('wall_s','memory_effective_rank')}
 if isinstance(v,list):return [canon(x) for x in v]
 return v

def main():
 p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--reference',required=True);p.add_argument('--candidate',required=True);p.add_argument('--exact',action='store_true');a=p.parse_args();sources={};inputs={};records=[]
 for tag in (a.reference,a.candidate):
  path=ROOT/f'experiments/results/token_language/{tag}.json';r=json.loads(path.read_text());assert r['status']=='completed';inputs[str(path.relative_to(ROOT))]=sha(path);sources.update(r['source_sha256']);records.append(r)
 for name,digest in sources.items():assert sha(ROOT/name)==digest,name
 r,c=records;assert r['presentations_total']==c['presentations_total']
 for key in r['args']:
  if key not in ('tag','resume','stop_after_step'):assert r['args'][key]==c['args'][key],key
 delta=c['curve'][-1]['dev_nll']-r['curve'][-1]['dev_nll'];ratio=c['train_wall_s_total']/r['train_wall_s_total'];exact=canon(r['curve'])==canon(c['curve']);passed=exact if a.exact else delta<=-.01 and ratio<=1.01
 sources[str(Path(__file__).relative_to(ROOT))]=sha(Path(__file__))
 d=dict(status='completed',battle='R1',gate='PASS' if passed else 'STOP',exact_recorded_learning=exact,final_nll_delta=delta,training_wall_ratio=ratio,targets=c['presentations_total'],credit_prediction=c['credit_prediction'],teacher_work=c['teacher_work'],source_sha256=sources,input_sha256=inputs,scope='Single-seed R1 development; off-mode requires exact learner trace except SVD rounding. Linear residual promotion requires≥0.01nats finalDEV gain vs retained fixed-admission recipe and≤1% extra complete CPU time. Same backward call, critic input/actor-credit derivatives detached as declared; actor clipping/Adam retained, predictor local elementwise-clippedSGD. Receipt MSE is on realized teacher samples, not an independent prediction test or future-update utility. Public validation sealed.')
 with (ROOT/f'experiments/results/diagnostics/{a.tag}.json').open('x') as f:json.dump(d,f,indent=2);f.write('\n')
 print(json.dumps({k:v for k,v in d.items() if k not in ('source_sha256','input_sha256')}),flush=True)
if __name__=='__main__':main()
