"""Credit is a native temporal event model, with sparse relationship addresses.
Shared credit-token projection -> hard temporal race units -> linear loss message.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from structural_sensor import StructuralSensor
from sleeping_machines.addressed_event_heads import AddressedEventHeads
from sleeping_machines.paired_route_credit import paired_route_credit

class RaceCreditSensor(StructuralSensor):
 def __init__(self,confidence):
  super().__init__(confidence,True);p=self.core.total_payload
  self.critic=AddressedEventHeads(sources=1,content_dim=2*p+12,classes=1,payload=8,depth=2,pool=2,heads=2)
  nn.init.zeros_(self.critic.embedding.weight);nn.init.zeros_(self.critic.head.weight);nn.init.zeros_(self.critic.head.bias);self.audit_probability=.25
 def fresh(self):
  st=super().fresh();st.update(credit_nodes={},credit_rng=torch.Generator().manual_seed(9127).get_state(),peak_credit_pairs=0);st['counts'].update(credit_events=0,credit_candidate_keys=0,credit_selected_updates=0,credit_counterfactual_values=0,credit_state_reads=0,critic_targets=0,exact_alternative_audits=0,critic_contrast_squared_error=0.,audited_contrast_squared_magnitude=0.);return st
 def credit_event(self,pair,t,features,kind,loss,audited,st):
  local=st['credit_nodes'].setdefault(pair,self.critic.new_state());before=(local.candidate_scores,local.selected_updates,local.counterfactual_values);token=torch.cat((features.detach(),features.new_tensor([kind,loss,audited])))
  # Credit races have their own random stream: they never consume actor noise.
  with torch.random.fork_rng():
   torch.set_rng_state(st['credit_rng']);prediction,_=self.critic.consume_event(0,t,token,local);st['credit_rng']=torch.get_rng_state()
  c=st['counts'];c['credit_events']+=1;c['credit_candidate_keys']+=local.candidate_scores-before[0];c['credit_selected_updates']+=local.selected_updates-before[1];c['credit_counterfactual_values']+=local.counterfactual_values-before[2];st['peak_credit_pairs']=max(st['peak_credit_pairs'],len(st['credit_nodes']));return prediction.squeeze()
 def cached_credit(self,pair,t,st):
  st['counts']['credit_state_reads']+=1
  if pair not in st['credit_nodes']:return 0.
  values,times=st['credit_nodes'][pair].contexts[0];at=torch.maximum(times.max(),times.new_tensor(t));value=self.critic.align(list(values.split(self.critic.payload)),times,at,self.critic.depth-1)
  return float(self.critic.head(value).detach().squeeze())
 def issue(self,r,st,generator,arm='exact',learning=False):
  assert 'target' not in r and r['query_id'] not in st['issued'];node,t,h=r['node'],r['time'],r['horizon'];target=self.context(node,t,st);bit=float(st['bits'].get(node,0));age=math.log1p(max(0,t-st['last'].get(node,t)));h=math.log1p(h);zero=target.new_tensor([bit,age,h]);q=self.query(torch.cat((target,zero)));candidates=[node]+[n for n in st['recent'] if n!=node][:self.candidates-1];contexts=[self.context(n,t,st) for n in candidates];aux=[target.new_tensor([float(st['bits'].get(n,0)),math.log1p(max(0,t-st['last'].get(n,t)))]) for n in candidates];edges=[]
  for source in candidates:
   count,same,transition,visits=st['edges'].get((node,source),[0.,0.,0.,0.]);edges.append(target.new_tensor([math.log1p(count),same/(count+1.),transition/(count+1.),count/(visits+1.)]))
  scores=torch.stack([(q*self.key(torch.cat((v,a)))).sum()/math.sqrt(len(q))+self.edge_score(e).squeeze() for v,a,e in zip(contexts,aux,edges)]).clamp(-12,12)
  if arm=='joint':scores=scores-scores.new_tensor([self.cached_credit((node,source),t,st) for source in candidates])
  pi=scores.softmax(0);noise=torch.empty(len(candidates),dtype=torch.float64).exponential_(generator=generator);winner=int((noise/scores.double().exp()).argmin());features=[torch.cat((target,v,zero,a,e)) for v,a,e in zip(contexts,aux,edges)];factual=self.sensor(features[winner]).squeeze();alternative=None;alt=None;audit=False;predicted=None
  if learning and len(candidates)>1:
   tg=st.setdefault('teacher_generator',torch.Generator().manual_seed(812));choices=[j for j in range(len(candidates)) if j!=winner];alternative=choices[int(torch.randint(len(choices),(1,),generator=tg))];coin=float(torch.rand((),generator=tg));audit=arm=='exact' or coin<self.audit_probability
   if audit:
    with torch.no_grad():alt=self.sensor(features[alternative]).squeeze()
  # Query tokens run only chosen/sampled credit producers, not all candidates.
  # During evaluation the joint policy reads acquired episode memory, then the
  # selected credit producer still evolves numerically without fitting parameters.
  if arm!='exact':
   selected=[winner]+([alternative] if alternative is not None else []);predicted={i:self.credit_event((node,candidates[i]),t,features[i],0.,0.,0.,st) for i in selected}
  c=st['counts'];c['queries']+=1;c['candidate_keys']+=len(candidates);c['discovery_context_reads']+=len(candidates)+1;c['relationship_reads']+=len(candidates);c['value_reads']+=1+int(audit);c['sensor_decodes']+=1+int(audit);c['exact_alternative_audits']+=int(audit)
  st['issued'][r['query_id']]=dict(time=t,logit=factual,probability=float(factual.detach().sigmoid()),pi=pi,winner=winner,alternative=alternative,alt=alt,q=1/max(1,len(candidates)-1),predicted=predicted,arm=arm,audit=audit,node=node,candidates=candidates,features=[f.detach() for f in features]);return st['issued'][r['query_id']]['probability']
 def feedback(self,r,st,learning):
  packet=st['issued'].pop(r['query_id']);assert packet['time']<r['time'];y=packet['logit'].new_tensor(float(r['target']));loss=F.binary_cross_entropy_with_logits(packet['logit'],y);base=loss.detach();pred=packet['predicted'];w=packet['winner'];j=packet['alternative'];critic_loss=loss.new_zeros(());actual=None
  if pred is not None:
   if learning:critic_loss=F.mse_loss(pred[w],base);st['counts']['critic_targets']+=1
   self.credit_event((packet['node'],packet['candidates'][w]),r['time'],packet['features'][w],1.,float(base),1.,st)
  if learning and j is not None:
   if packet['arm']=='exact':delta=F.binary_cross_entropy_with_logits(packet['alt'],y)-base
   else:
    estimate=(pred[j]-pred[w]).detach();delta=estimate
    if packet['audit']:
     actual=F.binary_cross_entropy_with_logits(packet['alt'],y);st['counts']['critic_contrast_squared_error']+=float((actual-base-estimate).detach().square());st['counts']['audited_contrast_squared_magnitude']+=float((actual-base).detach().square());delta=estimate+(actual-base-estimate)/self.audit_probability;critic_loss=critic_loss+F.mse_loss(pred[j],actual.detach());st['counts']['critic_targets']+=1;self.credit_event((packet['node'],packet['candidates'][j]),r['time'],packet['features'][j],1.,float(actual.detach()),1.,st)
   loss=loss+paired_route_credit(packet['pi'][None],torch.tensor([j]),loss.new_tensor([packet['q']]),delta[None])
  return loss+critic_loss,packet['probability']
 def detach(self,st):
  super().detach(st)
  for local in st['credit_nodes'].values():local.detach()
