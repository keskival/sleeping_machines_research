"""Connected future-loss critic for source selection and paid route credit.
Critic labels are revealed forecast losses, never target gradients/hidden edges.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from native_sensor import Sensor
from sleeping_machines.paired_route_credit import paired_route_credit

class LearnedSensor(Sensor):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs);p=self.core.total_payload
  # All routes share a critic and a summary of the candidate set. No isolated address heads.
  self.critic=nn.Sequential(nn.Linear(3*p+5,64),nn.Tanh(),nn.Linear(64,32),nn.Tanh(),nn.Linear(32,1))
  nn.init.zeros_(self.critic[-1].weight);nn.init.zeros_(self.critic[-1].bias)
  self.audit_probability=.25
 def fresh(self):
  st=super().fresh();st['counts'].update(discovery_context_reads=0,critic_candidates=0,critic_targets=0,exact_alternative_audits=0);return st
 def issue(self,r,st,generator,arm='exact',learning=False):
  assert 'target' not in r and r['query_id'] not in st['issued'];node,t,h=r['node'],r['time'],r['horizon'];target=self.context(node,t,st);bit=float(st['bits'].get(node,0));age=math.log1p(max(0,t-st['last'].get(node,t)));h=math.log1p(h);zero=target.new_tensor([bit,age,h]);q=self.query(torch.cat((target,zero)))
  candidates=[node]+[n for n in st['recent'] if n!=node][:self.candidates-1];contexts=[self.context(n,t,st) for n in candidates];aux=[target.new_tensor([float(st['bits'].get(n,0)),math.log1p(max(0,t-st['last'].get(n,t)))]) for n in candidates];summary=torch.stack(contexts).mean(0)
  scores=torch.stack([(q*self.key(torch.cat((v,a)))).sum()/math.sqrt(len(q)) for v,a in zip(contexts,aux)]).clamp(-12,12)
  predicted=None
  if arm!='exact':
   # Detach actor representations: the critic is supervised on issued context,
   # and cannot alter the actor representation by making loss labels easier.
   features=torch.stack([torch.cat((target,v,summary,zero,a)).detach() for v,a in zip(contexts,aux)])
   predicted=self.critic(features).squeeze(-1)
   if arm=='joint':scores=scores-predicted.detach()
   st['counts']['critic_candidates']+=len(candidates)
  pi=scores.softmax(0);noise=torch.empty(len(candidates),dtype=torch.float64).exponential_(generator=generator);winner=int((noise/scores.double().exp()).argmin());factual=self.sensor(torch.cat((target,contexts[winner],zero,aux[winner]))).squeeze();alternative=None;alt=None;audit=False
  if learning and len(candidates)>1:
   # Independent teaching RNG cannot perturb future factual race draws.
   tg=st.setdefault('teacher_generator',torch.Generator().manual_seed(812));choices=[j for j in range(len(candidates)) if j!=winner];alternative=choices[int(torch.randint(len(choices),(1,),generator=tg))];audit=(arm=='exact') or bool(torch.rand((),generator=tg)<self.audit_probability)
   if audit:
    with torch.no_grad():alt=self.sensor(torch.cat((target,contexts[alternative],zero,aux[alternative]))).squeeze()
  c=st['counts'];c['queries']+=1;c['candidate_keys']+=len(candidates);c['discovery_context_reads']+=len(candidates)+1;c['value_reads']+=1+int(audit);c['sensor_decodes']+=1+int(audit);c['exact_alternative_audits']+=int(audit)
  st['issued'][r['query_id']]=dict(time=t,logit=factual,probability=float(factual.detach().sigmoid()),pi=pi,winner=winner,alternative=alternative,alt=alt,q=1/max(1,len(candidates)-1),predicted=predicted,arm=arm,audit=audit)
  return st['issued'][r['query_id']]['probability']
 def feedback(self,r,st,learning):
  packet=st['issued'].pop(r['query_id']);assert packet['time']<r['time'];y=packet['logit'].new_tensor(float(r['target']));loss=F.binary_cross_entropy_with_logits(packet['logit'],y);base=loss.detach()
  if not learning:return loss,packet['probability']
  pred=packet['predicted'];w=packet['winner'];j=packet['alternative'];critic_loss=loss.new_zeros(())
  if pred is not None:
   critic_loss=F.mse_loss(pred[w],base);st['counts']['critic_targets']+=1
  if j is not None:
   if packet['arm']=='exact':delta=F.binary_cross_entropy_with_logits(packet['alt'],y)-base
   else:
    estimate=(pred[j]-pred[w]).detach();delta=estimate
    if packet['audit']:
     actual=F.binary_cross_entropy_with_logits(packet['alt'],y);delta=estimate+(actual-base-estimate)/self.audit_probability;critic_loss=critic_loss+F.mse_loss(pred[j],actual.detach());st['counts']['critic_targets']+=1
   loss=loss+paired_route_credit(packet['pi'][None],torch.tensor([j]),loss.new_tensor([packet['q']]),delta[None])
  return loss+critic_loss,packet['probability']
