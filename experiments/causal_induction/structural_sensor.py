"""Persistent causal observation evidence for shared integrated sensor routes.
Evidence is learned from public transitions; it is not an oracle edge teacher.
"""
import math
import torch
from torch import nn
from native_sensor import Sensor

class ResidualHead(nn.Module):
 def __init__(self,p,extra,confidence):
  super().__init__();self.p=p;self.base=nn.Parameter(torch.tensor(math.log(confidence/(1-confidence)),dtype=torch.float32));self.net=nn.Sequential(nn.Linear(2*p+5+extra,32),nn.Tanh(),nn.Linear(32,1));nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
 def forward(self,x):return (2*x[...,2*self.p]-1)*(self.base+self.net(x).squeeze(-1)).unsqueeze(-1) if x.ndim>1 else ((2*x[2*self.p]-1)*(self.base+self.net(x).squeeze())).reshape(1)

class StructuralSensor(Sensor):
 def __init__(self,confidence=.7,structural=True):
  super().__init__();self.structural=structural;self.sensor=ResidualHead(self.core.total_payload,4 if structural else 0,confidence)
  if structural:self.edge_score=nn.Linear(4,1,bias=False);nn.init.zeros_(self.edge_score.weight)
 def fresh(self):
  st=super().fresh();st.update(edges={},root_gap=None,last_root=None);st['counts'].update(discovery_context_reads=0,relationship_candidate_updates=0,relationship_reads=0);st['peak_edges']=0;return st
 def observe(self,r,st):
  node,t=r['node'],r['time']
  if r['exogenous']:
   if st['last_root'] is not None:
    gap=t-st['last_root']
    if gap>0:st['root_gap']=gap if st['root_gap'] is None else .9*st['root_gap']+.1*gap
   st['last_root']=t
  if self.structural:
   # Only already observed source transitions contribute. Per-event bounded work.
   period=st['root_gap'] or 1.;newbit=r['value'];oldbit=st['bits'].get(node,0)
   for source in st['recent']:
    if source==node:continue
    lag=max(0,t-st['last'][source])/period;key=(node,source);prior=st['edges'].get(key,[0.,0.,0.,0.]);weight=math.exp(-lag/.1);same=float(newbit==st['bits'][source]);transition=float(newbit!=oldbit)
    st['edges'][key]=[prior[0]+weight,prior[1]+weight*same,prior[2]+weight*transition,prior[3]+1.];st['counts']['relationship_candidate_updates']+=1
   st['peak_edges']=max(st['peak_edges'],len(st['edges']))
  super().observe(r,st)
 def issue(self,r,st,generator,arm='routed',learning=False):
  # Reuse the validated native route/feedback protocol. Edge evidence enters
  # exactly the candidate score and sensor decode, not labels or core weights.
  if not self.structural:
   before=st['counts']['candidate_keys'];p=super().issue(r,st,generator,arm,learning);st['counts']['discovery_context_reads']+=st['counts']['candidate_keys']-before+1;return p
  assert 'target' not in r and r['query_id'] not in st['issued'];node,t,h=r['node'],r['time'],r['horizon'];target=self.context(node,t,st);bit=float(st['bits'].get(node,0));age=math.log1p(max(0,t-st['last'].get(node,t)));h=math.log1p(h);zero=target.new_tensor([bit,age,h]);q=self.query(torch.cat((target,zero)));candidates=[node]+[n for n in st['recent'] if n!=node][:self.candidates-1]
  if arm=='self':candidates=[node]
  contexts=[self.context(n,t,st) for n in candidates];aux=[target.new_tensor([float(st['bits'].get(n,0)),math.log1p(max(0,t-st['last'].get(n,t)))]) for n in candidates];edges=[]
  for source in candidates:
   count,same,transition,visits=st['edges'].get((node,source),[0.,0.,0.,0.]);edges.append(target.new_tensor([math.log1p(count),same/(count+1.),transition/(count+1.),count/(visits+1.)]))
  scores=torch.stack([(q*self.key(torch.cat((v,a)))).sum()/math.sqrt(len(q))+self.edge_score(e).squeeze() for v,a,e in zip(contexts,aux,edges)]).clamp(-12,12);pi=scores.softmax(0);noise=torch.empty(len(candidates),dtype=torch.float64).exponential_(generator=generator);winner=int((noise/scores.double().exp()).argmin())
  def decode(index):return self.sensor(torch.cat((target,contexts[index],zero,aux[index],edges[index]))).squeeze()
  factual=decode(winner);alt=None;alternative=None
  if learning and len(candidates)>1:
   tg=st.setdefault('teacher_generator',torch.Generator().manual_seed(812));choices=[j for j in range(len(candidates)) if j!=winner];alternative=choices[int(torch.randint(len(choices),(1,),generator=tg))]
   with torch.no_grad():alt=decode(alternative)
  c=st['counts'];c['queries']+=1;c['candidate_keys']+=len(candidates);c['discovery_context_reads']+=len(candidates)+1;c['relationship_reads']+=len(candidates);c['value_reads']+=1+int(alt is not None);c['sensor_decodes']+=1+int(alt is not None)
  st['issued'][r['query_id']]=dict(time=t,logit=factual,probability=float(factual.detach().sigmoid()),pi=pi,alternative=alternative,alt=alt,q=1/max(1,len(candidates)-1));return st['issued'][r['query_id']]['probability']
