"""B5/R1 native source-address binding and integrated learning contract."""
import argparse,hashlib,json,resource,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'experiments/causal_induction'))
from source_addressed_world_credit import SourceAddressedWorldCredit
from world_credit_sensor import WorldCreditSensor
from world_credit_run import run

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args();torch.set_num_threads(1);begin=time.monotonic();torch.manual_seed(194)
 baseline=WorldCreditSensor(.8);child=SourceAddressedWorldCredit(.8);child.load_state_dict(baseline.state_dict())
 assert list(dict(child.named_parameters()))==list(dict(baseline.named_parameters()))
 st=child.fresh();g=torch.Generator().manual_seed(7)
 with torch.no_grad():
  child.observe(dict(kind='event',node=10,time=0.,value=1,exogenous=True),st)
  child.issue(dict(kind='query',node=30,time=.005,horizon=.2,query_id=0),st,g)
  first=st['issued'][0]['probability'];assert st['issued'][0]['world_source']==10
  child.observe(dict(kind='event',node=20,time=.01,value=0,exogenous=True),st)
  child.issue(dict(kind='query',node=30,time=.015,horizon=.2,query_id=1),st,g)
  assert st['issued'][0]['probability']==first and st['issued'][1]['world_source']==20
  child.feedback(dict(kind='feedback',time=.2,query_id=0,target=1),st)
  child.feedback(dict(kind='feedback',time=.3,query_id=1,target=0),st)
 assert not st['issued'] and set(st['credit_nodes'])=={(30,10),(30,20)}
 assert st['counts']['source_addressed_world_events']==4 and st['counts']['bank_writes']>0
 parent=ROOT/'experiments/results/causal_induction/curie_world_credit_20261012_v1_acquisition.json';d=json.loads(parent.read_text());record=d['data_records'][0];path=ROOT/record['path'];assert sha(path)==record['sha256']
 optimizer=torch.optim.Adam(child.parameters(),lr=.001);measured=run(child,path,'learned',optimizer,seed=194,max_queries=16)
 actor_grad=sum(float(p.grad.square().sum()) for n,p in child.named_parameters() if not n.startswith('critic.') and p.grad is not None)
 critic_grad=sum(float(p.grad.square().sum()) for n,p in child.named_parameters() if n.startswith('critic.') and p.grad is not None)
 assert measured['optimizer_updates']==2 and actor_grad>0 and critic_grad>0
 sources=dict(d['source_sha256'])
 for name in ['experiments/causal_induction/source_addressed_world_credit.py','experiments/causal_induction/SOURCE_ADDRESSED_WORLD_PROTOCOL.md',str(Path(__file__).relative_to(ROOT))]:sources[name]=sha(ROOT/name)
 result=dict(status='completed',battle='B5/R1',checks=dict(parameter_schema_preserved=True,overlapping_issue_source_binding=True,earlier_forecast_unchanged=True,finite_native_actor_world_training=True,actor_gradient_squared_norm=actor_grad,critic_gradient_squared_norm=critic_grad),rows=[measured],source_sha256=sources,input_sha256={str(parent.relative_to(ROOT)):sha(parent),record['path']:sha(path)},whole_wall_s=time.monotonic()-begin,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scope='Native child receipt/parameter contract and16target source integrated smoke; no utility comparison or long acquisition/deployment admission.')
 with (ROOT/f'experiments/results/diagnostics/{a.tag}.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(dict(checks=result['checks'],wall_s=result['whole_wall_s'])),flush=True)
if __name__=='__main__':main()
