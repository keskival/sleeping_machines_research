"""R1 teacher-admission gate: p1 exactness, then quality and paid replay work."""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments'))
def cleaned_curve(value):
 if isinstance(value,dict):return {k:cleaned_curve(v) for k,v in value.items() if k not in ('wall_s','replay_length','skipped_prefix')}
 if isinstance(value,list):return [cleaned_curve(v) for v in value]
 return value

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--reference',required=True);p.add_argument('--candidate',required=True);p.add_argument('--exact',action='store_true');a=p.parse_args()
 blobs={};sources={};inputs={}
 for key,name in [('reference',a.reference),('candidate',a.candidate)]:
  path=ROOT/f'experiments/results/token_language/{name}.json';d=json.loads(path.read_text());assert d['status']=='completed';blobs[key]=d;inputs[str(path.relative_to(ROOT))]=sha(path)
  for name,digest in d['source_sha256'].items():assert sha(ROOT/name)==digest,name;sources[name]=digest
 r,c=blobs['reference'],blobs['candidate'];assert r['presentations_total']==c['presentations_total']
 for key in r['args']:
  if key not in ('tag','resume','stop_after_step'):assert r['args'][key]==c['args'][key],key
 def curve(d):
  rows=cleaned_curve(d['curve'])
  for row in rows:row.pop('memory_effective_rank',None)
  return rows
 exact=curve(r)==curve(c)
 delta=c['curve'][-1]['dev_nll']-r['curve'][-1]['dev_nll'];ratio=c['train_wall_s_total']/r['train_wall_s_total']
 passed=exact if a.exact else delta<=.01 and ratio<1
 sources[str(Path(__file__).relative_to(ROOT))]=sha(Path(__file__))
 result=dict(status='completed',battle='R1',gate='PASS' if passed else 'STOP',exact_recorded_learning=exact,final_nll_delta=delta,training_wall_ratio=ratio,reference_training_s=r['train_wall_s_total'],candidate_training_s=c['train_wall_s_total'],teacher_work=c['teacher_work'],source_sha256=sources,input_sha256=inputs,scope='Single-seed development. Probability1 requires exact recorded learning except nondifferentiable SVD rounding. Sparse teacher gate allows at most0.01nats final DEV degradation and requires lower complete CPU training time; replay work counted directly, not full FLOPs or ASIC energy. Clipping/Adam can change trajectories despite conditional unbiased raw teacher gradients.')
 with (ROOT/f'experiments/results/diagnostics/{a.tag}.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(result),flush=True)
if __name__=='__main__':main()
