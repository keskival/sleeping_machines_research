"""B5/R1 frozen source-development audit: can any decoded route beat persistence?"""
import argparse,hashlib,json,sys,time,resource
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/causal_induction'))
from world_credit_sensor import WorldCreditSensor
from world_credit_run import run

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args();torch.set_num_threads(1);start=time.monotonic()
 source=ROOT/'experiments/results/causal_induction/curie_world_credit_20261012_v1_acquisition.json';d=json.loads(source.read_text());assert d['status']=='completed'
 for p,h in d['source_sha256'].items():assert sha(ROOT/p)==h,p
 checkpoint=ROOT/d['checks']['checkpoint'];assert sha(checkpoint)==d['checks']['checkpoint_sha256']
 saved=torch.load(checkpoint,map_location='cpu',weights_only=False);model=WorldCreditSensor(saved['confidence']);model.load_state_dict(saved['model']);model.eval()
 before={n:p.detach().clone() for n,p in model.named_parameters()};inputs={str(source.relative_to(ROOT)):sha(source),str(checkpoint.relative_to(ROOT)):sha(checkpoint)}
 key_inputs=[];edge_inputs=[];factual_inputs=[];issued=[]
 hooks=[model.key.register_forward_pre_hook(lambda m,x:key_inputs.append(x[0].detach().clone())),model.edge_score.register_forward_pre_hook(lambda m,x:edge_inputs.append(x[0].detach().clone())),model.sensor.register_forward_pre_hook(lambda m,x:factual_inputs.append(x[0].detach().clone()))]
 original=model.issue
 def issue(r,st,generator,arm='exact',learning=False):
  assert not learning;key_inputs.clear();edge_inputs.clear();factual_inputs.clear()
  probability=original(r,st,generator,arm,False);assert len(factual_inputs)==1 and len(key_inputs)==len(edge_inputs)
  x=factual_inputs[0];p=model.core.total_payload
  # Reuse already-read candidate inputs; no new bank reads or RNG draws.
  features=torch.stack([torch.cat((x[:p],key[:p],x[2*p:2*p+3],key[p:],edge)) for key,edge in zip(key_inputs,edge_inputs)])
  logits=model.sensor(features).reshape(-1);factual=st['issued'][r['query_id']]['logit']
  assert torch.allclose(logits[st['issued'][r['query_id']]['winner']],factual,atol=2e-6,rtol=2e-6)
  bit=int(st['bits'].get(r['node'],0));prediction=(logits>=0).to(torch.int).tolist()
  issued.append(dict(query_id=r['query_id'],bit=bit,predictions=prediction,route_logits=logits.tolist(),factual_probability=probability))
  return probability
 model.issue=issue;rows=[]
 # Fixed four source systems, first64 queries each. No target cohort selection.
 for index,record in enumerate(d['data_records'][:4]):
  assert record['split']=='train';path=ROOT/record['path'];assert sha(path)==record['sha256'];inputs[record['path']]=sha(path);issued.clear()
  measured=run(model,path,'exact',optimizer=None,seed=194+index,max_queries=64)
  by_id={r['query_id']:r for r in measured['issued_trace']};n=any_flip=beneficial=harmful=need_flip=0;margins=[];gaps=[]
  for packet in issued:
   y=by_id[packet['query_id']]['target'];changes=[v!=packet['bit'] for v in packet['predictions']];n+=1;any_flip+=int(any(changes));need_flip+=int(y!=packet['bit']);beneficial+=int(any(changes) and y!=packet['bit']);harmful+=int(any(changes) and y==packet['bit'])
   margins.extend([(2*packet['bit']-1)*v for v in packet['route_logits']]);gaps.append(max(packet['route_logits'])-min(packet['route_logits']))
  rows.append(dict(path=record['path'],targets=n,persistence_errors=need_flip,queries_with_any_answer_changing_route=any_flip,queries_with_any_beneficial_route=beneficial,queries_with_any_harmful_route=harmful,min_persistence_margin=min(margins),mean_route_logit_range=sum(gaps)/n,extra_decoder_routes=sum(len(r['predictions']) for r in issued),frozen_actor_nll=measured['nll'],frozen_world_nll=measured['world_nll'],wall_s=measured['wall_s']))
 for hook in hooks:hook.remove()
 assert all(torch.equal(p,before[n]) for n,p in model.named_parameters())
 result=dict(status='completed',battle='B5/R1',rows=rows,totals={k:sum(r[k] for r in rows) for k in ['targets','persistence_errors','queries_with_any_answer_changing_route','queries_with_any_beneficial_route','queries_with_any_harmful_route','extra_decoder_routes']},input_sha256=inputs,source_sha256={**d['source_sha256'],str(Path(__file__).relative_to(ROOT)):sha(Path(__file__))},whole_wall_s=time.monotonic()-start,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scope='Frozen acquired source194 actor/world; four fixed source TRAIN systems, first64targets, fresh numerical memory with actual delayed outcomes and no optimizer. All already-discovered candidate heads decoded as paid diagnostic; this is not sparse inference or a deployable target-conditioned selector. Any-beneficial-route counts use labels only after delivery. No new fitting, target selection, public test or full-family limit.')
 out=ROOT/f'experiments/results/diagnostics/{a.tag}.json'
 with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(dict(totals=result['totals'],whole_wall_s=result['whole_wall_s'])),flush=True)
if __name__=='__main__':main()
