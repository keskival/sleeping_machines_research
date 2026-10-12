"""Future-trained bounded step sizes and momentum on event credit responsibilities."""
import math
import torch
from event_parameter_credit import EventCredit

class EventOptimizerCredit(EventCredit):
    def __init__(self,width,named,queries=4):
        super().__init__(width,named,queries);self.step_logits=torch.nn.ParameterDict({k:torch.nn.Parameter(torch.tensor(math.log(.001/.002))) for k in self.decoders});self.momentum_logit=torch.nn.Parameter(torch.tensor(math.log(.9/.1)))
    def propose(self,named,direction,velocity=None):
        velocity={} if velocity is None else velocity;pieces=direction.split([named[n].numel() for n in self.names]);beta=torch.sigmoid(self.momentum_logit);steps={};next_velocity={}
        for name,piece in zip(self.names,pieces):
            g=piece.reshape_as(named[name]);v=beta*velocity.get(name,torch.zeros_like(g)).detach()+(1-beta)*g;next_velocity[name]=v.detach().clone()
            scale=v.square().mean().clamp_min(.01).sqrt();step=.003*torch.sigmoid(self.step_logits[self.key(name)])*v/scale;steps[name]=step
        norm=torch.stack([s.square().sum() for s in steps.values()]).sum().clamp_min(1e-24).sqrt();factor=(.2/norm).clamp_max(1)
        return {n:p.detach()-factor*steps[n] if n in steps else p.detach().clone() for n,p in named.items()},next_velocity
