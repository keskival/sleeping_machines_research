"""Shared integrated temporal race core with public-only future sensor queries.
No address embedding or generator structure: addresses select numerical state.
"""
import math,sys
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from sleeping_machines.addressed_event_heads import AddressedEventHeads
from sleeping_machines.paired_route_credit import paired_route_credit

class Sensor(nn.Module):
 def __init__(self,payload=8,depth=2,pool=2,heads=2,candidates=8):
  super().__init__();self.core=AddressedEventHeads(sources=1,content_dim=2,classes=1,payload=payload,depth=depth,pool=pool,heads=heads);self.core.head=nn.Identity();p=self.core.total_payload;self.candidates=candidates
  # Address identities are not parameters. Shared operators act on each live node.
  nn.init.zeros_(self.core.embedding.weight)
  self.key=nn.Linear(p+2,p,bias=False);self.query=nn.Linear(p+3,p,bias=False)
  self.sensor=nn.Sequential(nn.Linear(2*p+5,32),nn.Tanh(),nn.Linear(32,1))
 def fresh(self):return dict(nodes={},bits={},last={},recent=[],issued={},counts=dict(events=0,queries=0,candidate_keys=0,value_reads=0,sensor_decodes=0,selected_updates=0,counterfactual_unit_values=0),peak_nodes=0)
 def observe(self,r,st):
  node=r['node'];local=st['nodes'].setdefault(node,self.core.new_state());before=(local.selected_updates,local.counterfactual_values)
  self.core.consume_event(0,r['time'],[r['value'],float(r['exogenous'])],local)
  st['bits'][node]=r['value'];st['last'][node]=r['time'];st['recent']=[node]+[n for n in st['recent'] if n!=node][:self.candidates-1]
  c=st['counts'];c['events']+=1;c['selected_updates']+=local.selected_updates-before[0];c['counterfactual_unit_values']+=local.counterfactual_values-before[1];st['peak_nodes']=max(st['peak_nodes'],len(st['nodes']))
 def context(self,node,t,st):
  if node not in st['nodes']:return self.core.embedding.weight.new_zeros(self.core.total_payload)
  values,times=st['nodes'][node].contexts[0]
  # Read only past observed state; computational arrival may lag physical time.
  at=torch.maximum(times.max(),times.new_tensor(t))
  return self.core.align(list(values.split(self.core.payload)),times,at,self.core.depth-1)
 def issue(self,r,st,generator,arm='routed',learning=False):
  assert 'target' not in r and r['query_id'] not in st['issued'];node,t,h=r['node'],r['time'],r['horizon'];target=self.context(node,t,st);bit=float(st['bits'].get(node,0));age=math.log1p(max(0,t-st['last'].get(node,t)));h=math.log1p(h)
  zero=target.new_tensor([bit,age,h]);q=self.query(torch.cat((target,zero)))
  candidates=[node]+[n for n in st['recent'] if n!=node][:self.candidates-1]
  if arm=='self':candidates=[node]
  contexts=[self.context(n,t,st) for n in candidates];aux=[target.new_tensor([float(st['bits'].get(n,0)),math.log1p(max(0,t-st['last'].get(n,t)))]) for n in candidates]
  scores=torch.stack([(q*self.key(torch.cat((v,a)))).sum()/math.sqrt(len(q)) for v,a in zip(contexts,aux)]).clamp(-12,12);pi=scores.softmax(0)
  noise=torch.empty(len(candidates),dtype=torch.float64).exponential_(generator=generator);winner=int((noise/scores.double().exp()).argmin())
  def decode(index):return self.sensor(torch.cat((target,contexts[index],zero,aux[index]))).squeeze()
  factual=decode(winner);alt=None;alternative=None
  if learning and len(candidates)>1:
   choices=[j for j in range(len(candidates)) if j!=winner];alternative=choices[int(torch.randint(len(choices),(1,),generator=generator))]
   with torch.no_grad():alt=decode(alternative)
  st['counts']['queries']+=1;st['counts']['candidate_keys']+=len(candidates);st['counts']['value_reads']+=1+int(alt is not None);st['counts']['sensor_decodes']+=1+int(alt is not None)
  st['issued'][r['query_id']]=dict(time=t,logit=factual,probability=float(factual.detach().sigmoid()),pi=pi,alternative=alternative,alt=alt,q=1/max(1,len(candidates)-1))
  return st['issued'][r['query_id']]['probability']
 def feedback(self,r,st,learning):
  packet=st['issued'].pop(r['query_id']);assert packet['time']<r['time'];y=packet['logit'].new_tensor(float(r['target']));loss=F.binary_cross_entropy_with_logits(packet['logit'],y)
  if learning and packet['alt'] is not None:
   delta=F.binary_cross_entropy_with_logits(packet['alt'],y)-loss.detach();loss=loss+paired_route_credit(packet['pi'][None],torch.tensor([packet['alternative']]),loss.new_tensor([packet['q']]),delta[None])
  return loss,packet['probability']
 def detach(self,st):
  assert not st['issued'],'Do not mutate parameters while issued differentiable predictions remain pending'
  for node in st['nodes'].values():node.detach()
