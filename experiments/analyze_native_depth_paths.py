"""R1 frozen D8 stage message interventions; guarded inference only."""
import argparse,json,sys,time
from pathlib import Path
from types import SimpleNamespace,FunctionType
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
import bernoulli_teacher_token_language as driver
import sleeping_machines.ordered_token_episodes as episode
from public_token_scoring import score

def main():
 p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args();torch.set_num_threads(1)
 run='aws_native_deep_d8_8m_20261013_s6_v1';base=ROOT/f'experiments/results/token_language/{run}';r=json.loads(base.with_suffix('.json').read_text());assert r['status']=='completed'
 sources=dict(r['source_sha256'])
 for name,digest in sources.items():assert driver.sha(ROOT/name)==digest,name
 args=SimpleNamespace(**r['args']);train=driver.load_tokens(args.train_file);dev=driver.load_tokens(args.dev_file);counts=driver.LaneTokens(train,0,args.train_tokens,args.lanes).counts();order=torch.argsort(counts,descending=True,stable=True)
 torch.manual_seed(args.seed);model=driver.Model(args,order,counts);ck=torch.load(base.with_suffix('.pt'),map_location='cpu',weights_only=False);model.load_state_dict(ck['model']);del ck
 original=episode.original;step=original.__globals__['sparse_counterfactual_layer'];rows=[]
 for cut in [None]+list(range(args.depth)):
  calls=0;stats=[dict(count=0,input_square=0.,output_square=0.,residual_square=0.) for _ in range(args.depth)]
  def instrumented(*values,**kw):
   nonlocal calls
   d=calls%args.depth;calls+=1;out=step(*values,**kw);x,y=values[0],out[0];s=stats[d];s['count']+=x.numel();s['input_square']+=float(x.double().square().sum());s['output_square']+=float(y.double().square().sum());s['residual_square']+=float((y-x).double().square().sum())
   if d==cut:out=(torch.zeros_like(y),*out[1:])
   return out
  ns=dict(original.__globals__);ns['sparse_counterfactual_layer']=instrumented
  fn=FunctionType(original.__code__,ns,original.__name__,original.__defaults__,original.__closure__);fn.__kwdefaults__=original.__kwdefaults__;episode.original=fn
  began=time.monotonic()
  try:scored=score(model,dev,args,driver,25165824,8192,1024,8)
  finally:episode.original=original
  for s in stats:
   for key in ('input','output','residual'):s[key+'_rms']=(s.pop(key+'_square')/s['count'])**.5
  rows.append(dict(cut_stage=cut,scored=scored,stage_stats=stats,wall_s=time.monotonic()-began))
  print(json.dumps(dict(cut=cut,nll=scored['nll'])),flush=True)
 expected=json.loads((ROOT/'experiments/results/diagnostics/aws_native_deep_d8_8m_quality_20261013_s6_v1.json').read_text())['rows'][1]['scored']['nll'];assert abs(rows[0]['scored']['nll']-expected)<1e-12
 for row in rows:row['nll_change']=row['scored']['nll']-rows[0]['scored']['nll']
 for name in ['experiments/analyze_native_depth_paths.py','experiments/public_token_scoring.py']:sources[name]=driver.sha(ROOT/name)
 inputs={str(p.relative_to(ROOT)):driver.sha(p) for p in [base.with_suffix('.json'),base.with_suffix('.pt'),ROOT/'experiments/results/diagnostics/aws_native_deep_d8_8m_quality_20261013_s6_v1.json']}
 with (ROOT/f'experiments/results/diagnostics/{a.tag}.json').open('x') as f:json.dump(dict(status='completed',battle='R1',rows=rows,source_sha256=sources,input_sha256=inputs,scope='Frozen D8 final weights,8192 reusedDEV targets/1024history/seed6. Cuts zero each selected stage outgoing mixed message at every token; clocks and private writes remain computed. Distribution-changing necessity diagnostic, not retrained layer removal, pure memory attribution or sufficiency. RMS is not Jacobian conditioning. No fit/public scoring.'),f,indent=2)
if __name__=='__main__':main()
