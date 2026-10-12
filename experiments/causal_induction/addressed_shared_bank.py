"""Public relative-address semantics for independent native bank access.
No learned node-ID table and no hidden generating edges.
"""
import math
import torch
from torch import nn
from native_sensor import Sensor
from task_taught_shared_bank import TaskTaughtSharedBank


class AddressedSharedBank(TaskTaughtSharedBank):
 def __init__(self, confidence, mode='reciprocal'):
  super().__init__(confidence, mode)
  self.bank_relation=nn.Linear(5,1,bias=False)
  self.critic.bank_relation=nn.Linear(5,1,bias=False)
  nn.init.zeros_(self.bank_relation.weight)
  nn.init.zeros_(self.critic.bank_relation.weight)

 def fresh(self):
  st=super().fresh();st['bank_by_node']={};return st

 @staticmethod
 def endpoints(address):
  return (address[1],None) if address[0]=='F' else address[1]

 def write_record(self,address,role,value,arrival,st):
  super().write_record(address,role,value,arrival,st)
  for node in set(n for n in self.endpoints(address) if n is not None):
   local=st['bank_by_node'].get(node,[])
   st['bank_by_node'][node]=[address]+[a for a in local if a!=address]

 def read_bank(self,role,query,t,st,reader=None):
  reader=reader if reader is not None else st['credit_reader']
  visible=[];seen=set()
  # Addressed access is a supported public operation, not a parent oracle.
  candidates=[]
  for node in reader:candidates.extend(st['bank_by_node'].get(node,[]))
  candidates.extend(st['bank_order'])
  for address in candidates:
   st['counts']['bank_discovery_records']+=1
   if address in seen:continue
   seen.add(address)
   eligible=[r for r in st['bank'][address] if r['arrival']<=t]
   if eligible:visible.append((address,eligible[-1]))
   if len(visible)==self.bank_limit:break
  st.setdefault('bank_last_candidates',{})[role]=[a for a,_ in visible]
  if not visible:return query.new_zeros(query.shape)
  owner=self if role=='F' else self.critic
  q=owner.bank_query(query);transported=[];relations=[]
  first=reader[0];second=reader[1] if len(reader)>1 else None
  for address,r in visible:
   source_core=self.core if r['role']=='F' else self.critic
   arrival=torch.full((source_core.heads,),r['arrival'],dtype=torch.float64)
   value=source_core.align(list(r['value'].split(source_core.payload)),arrival,
                           arrival.new_tensor(t),source_core.depth-1)
   stop=role=='C' or (self.mode=='untaught' and r['role']=='C')
   transported.append(value.detach() if stop else value)
   target,source=self.endpoints(address)
   relations.append([float(r['role']=='C'),float(target==first),
                     float(source==first),float(second is not None and target==second),
                     float(second is not None and source==second)])
  values=torch.stack(transported)
  scores=(owner.bank_key(values)*q).sum(-1)/math.sqrt(query.numel())
  scores=scores+owner.bank_relation(query.new_tensor(relations)).squeeze(-1)
  scores=scores.clamp(-12,12)
  core=self.core if role=='F' else self.critic
  if role=='C':
   with torch.random.fork_rng():
    torch.set_rng_state(st['credit_rng']);selected,_,winner=core.race(scores,values)
    st['credit_rng']=torch.get_rng_state()
  else:selected,_,winner=core.race(scores,values)
  address,r=visible[int(winner)];assert r['arrival']<=t
  st['bank_read_log']=(st['bank_read_log']+[(role,address,r['version'],float(t),r['arrival'])])[-64:]
  c=st['counts'];c['bank_scored_keys']+=len(visible);c['bank_context_reads']+=len(visible)
  c['bank_value_reads']+=1;c['bank_counterfactual_values']+=len(visible) if self.training else 0
  if role=='F' and r['role']=='C':c['forward_credit_reads']+=1
  if role=='C' and r['role']=='F':c['critic_forward_reads']+=1
  return selected

 def context(self,node,t,st):
  value=Sensor.context(self,node,t,st)
  extra=self.read_bank('F',value,t,st,reader=(node,))
  if self.mode in ('separate','read_only'):return value
  return value+.1*extra.tanh()

 def credit_event(self,pair,t,features,kind,loss,audited,st):
  previous=st.get('credit_reader');st['credit_reader']=pair
  try:return super().credit_event(pair,t,features,kind,loss,audited,st)
  finally:
   if previous is None:st.pop('credit_reader',None)
   else:st['credit_reader']=previous
