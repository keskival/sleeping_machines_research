"""R1 frozen final-weight error analysis under public-style history on DEV only."""
import argparse,json,statistics,sys,time
from pathlib import Path
from types import SimpleNamespace
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'))
import window_replay_token_language as driver
from public_token_scoring import score

def main():
 p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args();torch.set_num_threads(1)
 gate_path=ROOT/'experiments/results/diagnostics/aws_bernoulli_teacher_8m_20261013_s6_v2_quality_priority_v2.json';gate=json.loads(gate_path.read_text());assert gate['status']=='completed'
 selected='8m' if gate['accepted'] else '2m';sources={};inputs={str(gate_path.relative_to(ROOT)):driver.sha(gate_path)};rows=[]
 for size,version in (('512k','v1'),(selected,'v2')):
  name=f'aws_bernoulli_teacher_{size}_20261013_s6_{version}';base=ROOT/f'experiments/results/token_language/{name}';r=json.loads(base.with_suffix('.json').read_text());assert r['status']=='completed'
  for name,digest in r['source_sha256'].items():assert driver.sha(ROOT/name)==digest,name;sources[name]=digest
  inputs[str(base.with_suffix('.json').relative_to(ROOT))]=driver.sha(base.with_suffix('.json'));checkpoint=base.with_suffix('.pt');inputs[str(checkpoint.relative_to(ROOT))]=driver.sha(checkpoint)
  args=SimpleNamespace(**r['args']);train=driver.load_tokens(args.train_file);dev=driver.load_tokens(args.dev_file);counts=driver.LaneTokens(train,0,args.train_tokens,args.lanes).counts();order=torch.argsort(counts,descending=True,stable=True)
  torch.manual_seed(args.seed);model=driver.Model(args,order,counts);ck=torch.load(checkpoint,map_location='cpu',weights_only=False);assert ck['step']==args.steps;model.load_state_dict(ck['model']);del ck
  original=driver.evaluate(model,driver.interval_tensor(dev,args.dev_offset,args.dev_tokens,args.eval_lanes),args);assert original==r['curve'][-1]['dev_nll']
  metrics=[dict(rank_low=lo,rank_high=hi,count=0,loss_sum=0.,zero_feature_loss_sum=0.) for lo,hi in ((0,2000),(2000,10000),(10000,30000),(30000,50257))]
  real=model.readout.nll
  def audited(features,targets):
   factual=real(features,targets);zero=real(torch.zeros_like(features),targets);ranks=model.readout.token_to_rank[targets.flatten()]
   for bucket in metrics:
    mask=(ranks>=bucket['rank_low'])&(ranks<bucket['rank_high']);bucket['count']+=int(mask.sum());bucket['loss_sum']+=float(factual.reshape(-1)[mask].double().sum());bucket['zero_feature_loss_sum']+=float(zero.reshape(-1)[mask].double().sum())
   return factual
  model.readout.nll=audited;start=time.monotonic()
  try:scored=score(model,dev,args,driver,25165824,8192,1024,8)
  finally:model.readout.nll=real
  for bucket in metrics:
   bucket['nll']=None if not bucket['count'] else bucket['loss_sum']/bucket['count'];bucket['feature_gain']=None if not bucket['count'] else (bucket['zero_feature_loss_sum']-bucket['loss_sum'])/bucket['count'];bucket['total_nll_contribution']=bucket['loss_sum']/8192;bucket['target_fraction']=bucket['count']/8192
  assert sum(x['count'] for x in metrics)==8192
  assert abs(sum(x['loss_sum'] for x in metrics)-scored['loss_sum'])<1e-8
  rows.append(dict(acquisition=size,scored=scored,buckets=metrics,tail_widths=[tail[0].out_features for tail in model.readout.output.tail],diagnostic_s=time.monotonic()-start))
 sources[str(Path(__file__).relative_to(ROOT))]=driver.sha(Path(__file__));sources['experiments/public_token_scoring.py']=driver.sha(ROOT/'experiments/public_token_scoring.py')
 d=dict(status='completed',battle='R1',rows=rows,source_sha256=sources,input_sha256=inputs,scope='Frozen final weights, seed6, same8192 previously used transfer-DEV tokens scored with disjoint1024-token histories and all shifted targets; no EOS reset, no public validation or checkpoint selection. Rank bins fitted from fixed initial TRAIN prior. Zero-feature intervention distinguishes readout intercepts from all factual input/state features, not persistent history alone. Tail widths are implementation capacity, not proof of a bottleneck. Diagnostic wall includes extra readout scoring; candidate repair requires integrated fitting comparison.')
 with (ROOT/f'experiments/results/diagnostics/{a.tag}.json').open('x') as f:json.dump(d,f,indent=2);f.write('\n')
 print(json.dumps(rows),flush=True)
if __name__=='__main__':main()
