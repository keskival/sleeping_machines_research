"""Native credit learns route improvement relative to a shared issued baseline.
Binary proper-loss control variate removes outcome variation common to routes.
"""
import torch
from torch.nn import functional as F
from race_credit_sensor import RaceCreditSensor
from sleeping_machines.paired_route_credit import paired_route_credit

class CenteredRaceCredit(RaceCreditSensor):
 def issue(self,r,st,generator,arm='credit',learning=False):
  p=super().issue(r,st,generator,arm,learning);bit=float(st['bits'].get(r['node'],0));st['issued'][r['query_id']]['anchor_logit']=((2*bit-1)*self.sensor.base).detach();return p
 def feedback(self,r,st,learning):
  packet=st['issued'].pop(r['query_id']);assert packet['time']<r['time'];y=packet['logit'].new_tensor(float(r['target']));loss=F.binary_cross_entropy_with_logits(packet['logit'],y);base=loss.detach();anchor=F.binary_cross_entropy_with_logits(packet['anchor_logit'],y).detach();center=base-anchor;pred=packet['predicted'];w=packet['winner'];j=packet['alternative'];critic_loss=loss.new_zeros(())
  if pred is not None:
   if learning:critic_loss=F.mse_loss(pred[w],center);st['counts']['critic_targets']+=1
   self.credit_event((packet['node'],packet['candidates'][w]),r['time'],packet['features'][w],1.,float(center),1.,st)
  if learning and j is not None:
   if packet['arm']=='exact':delta=F.binary_cross_entropy_with_logits(packet['alt'],y)-base
   else:
    estimate=(pred[j]-pred[w]).detach();delta=estimate
    if packet['audit']:
     actual=F.binary_cross_entropy_with_logits(packet['alt'],y);st['counts']['critic_contrast_squared_error']+=float((actual-base-estimate).detach().square());st['counts']['audited_contrast_squared_magnitude']+=float((actual-base).detach().square());delta=estimate+(actual-base-estimate)/self.audit_probability;critic_loss=critic_loss+F.mse_loss(pred[j],actual.detach()-anchor);st['counts']['critic_targets']+=1;self.credit_event((packet['node'],packet['candidates'][j]),r['time'],packet['features'][j],1.,float(actual.detach()-anchor),1.,st)
   loss=loss+paired_route_credit(packet['pi'][None],torch.tensor([j]),loss.new_tensor([packet['q']]),delta[None])
  return loss+critic_loss,packet['probability']
