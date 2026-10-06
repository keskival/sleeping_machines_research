"""Sequential actual-driver diagonal parity, causal endpoints and resume checks."""
import argparse,gc,hashlib,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
import horizon_token_language_engine as original
import split_horizon_token_language_engine as split

def equal(x,y):
 if isinstance(x,torch.Tensor):assert torch.equal(x,y)
 elif isinstance(x,dict):
  assert x.keys()==y.keys()
  for k in x:equal(x[k],y[k])
 elif isinstance(x,(list,tuple)):
  assert len(x)==len(y)
  for a,b in zip(x,y):equal(a,b)
 else:assert x==y
FIELDS=('model','optimizer','state','generator','alternative_generator','local_generator','position_generator','site_generator','torch_rng','cursor','step','total_presentations','writes')

def main():
 p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args()
 out=ROOT/f'experiments/results/diagnostics/{a.tag}.json';assert not out.exists()
 folder=ROOT/'experiments/results/token_language'
 base=['--readout-init','frequency','--input-init','balanced','--steps','4','--eval-every','1','--train-tokens','512','--dev-tokens','128','--payload','8','--depth','2','--heads','1','--pool','2','--lanes','2','--eval-lanes','2','--chunk','16','--future-site','uniform','--future-score-positions','2','--future-every','1']
 provenance={};records={}
 def run(label,engine,factual,alternative=None,extra=()):
  tag=a.tag+'_'+label;path=folder/(tag+'.json');assert not path.exists() and not path.with_suffix('.pt').exists()
  args=['--credit-window',str(factual)]
  if alternative is not None:args+=['--future-credit-window',str(alternative)]
  sys.argv=['driver','--tag',tag,*base,*args,*extra];engine.main();gc.collect()
  if not path.exists():path=path.with_suffix('.partial.json')
  result=json.loads(path.read_text());checkpoint=torch.load(folder/(tag+'.pt'),weights_only=False)
  for row in result['curve'][1:]:
   teacher=row['future_write_teacher'];token=teacher['site'][0];window=factual if alternative is None else alternative
   n=16;end=min(n,(token//window+1)*window)
   assert teacher['credit_end']==end
   assert all(token<=i<end for i in teacher['scoring_positions'])
   assert teacher['utility_weight']==teacher['inverse_site_probability']*(end-token)/n
  provenance[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
  records[label]=dict(status=result['status'],targets=result['presentations_total'],final_dev=result['curve'][-1]['dev_nll'])
  return checkpoint
 for horizon in (4,16):
  old=run(f'original{horizon}',original,horizon)
  default=run(f'default{horizon}',split,horizon)
  diagonal=run(f'diagonal{horizon}',split,horizon,horizon)
  for field in FIELDS:equal(old[field],default[field]);equal(old[field],diagonal[field])
  assert len(old['curve'])==len(default['curve'])==len(diagonal['curve'])
  for x,y,z in zip(old['curve'],default['curve'],diagonal['curve']):
   for field in ('step','train_nll','dev_nll','future_write_teacher'):assert (field in x)==(field in y)==(field in z);equal(x.get(field),y.get(field));equal(x.get(field),z.get(field))
  del old,default,diagonal;gc.collect()
 for factual,alternative in ((4,16),(16,4)):
  label=f'f{factual}a{alternative}';full=run(label,split,factual,alternative)
  partial=run(label+'_partial',split,factual,alternative,['--stop-after-step','2']);del partial;gc.collect()
  resumed=run(label+'_resumed',split,factual,alternative,['--resume',str(folder/(a.tag+'_'+label+'_partial.pt'))])
  for field in FIELDS:equal(full[field],resumed[field])
  assert len(full['curve'])==len(resumed['curve'])
  for x,y in zip(full['curve'],resumed['curve']):
   for field in ('step','train_nll','dev_nll','future_write_teacher'):assert (field in x)==(field in y);equal(x.get(field),y.get(field))
  del full,resumed;gc.collect()
 result=dict(status='completed',records=records,input_sha256=provenance,exact_fields=list(FIELDS),diagonal_parity=True,off_diagonal_resume=True,causal_endpoints=True,importance_weight_contract=True,scope='Tiny actual-driver learning contracts; no benchmark quality claim. Factual gradient-cut and cross-partition feature witnesses are separate required checks.')
 result['producer_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
