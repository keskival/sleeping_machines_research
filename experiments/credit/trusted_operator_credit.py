"""Future-trained trust region in actual parameter-step space, with independent anchor momentum."""
import math
import torch
from operator_event_credit import EventOptimizerCredit as OperatorCredit,local_vjp

class EventOptimizerCredit(OperatorCredit):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.trust=torch.nn.Linear(64,1);torch.nn.init.zeros_(self.trust.weight);torch.nn.init.constant_(self.trust.bias,math.log(.1/1.9))
    def forward(self,p,forced=None,pool=None):
        full=super().forward(p,forced,pool);ids=self.last_selected;Q=len(ids);length=len(p['cotangent']);message=p['cotangent'][ids];anchors={};operators=p['operators'];depth=len(p['states'])-1
        for layer in range(depth-1,-1,-1):
            prefix=f'layers.{layer}.';local={n[len(prefix):]:v for n,v in operators.items() if n.startswith(prefix)};message,gradients=local_vjp(p['inputs'][prefix+'write.weight'][ids],p['inputs'][prefix+'read.weight'][ids],p['dt'][ids],local,message);anchors.update({prefix+n:length*g for n,g in gradients.items()})
        anchors['gap.weight']=length*message.T@p['inputs']['gap.weight'][ids]/Q;anchors['gap.bias']=length*message.mean(0);anchors['embed.weight']=message.new_zeros(operators['embed.weight'].shape).index_add(0,p['marks'][ids],message)*length/Q
        self.anchor_direction=torch.cat([anchors[n].flatten() for n in self.names]);self.radius=2*torch.sigmoid(self.trust(self.local_contexts.mean((0,1))).squeeze());return full
    def propose(self,named,direction,velocity=None):
        state={} if velocity is None else velocity;full,full_velocity=super().propose(named,direction,state.get('full'));anchor_velocity={};anchor_steps={};pieces=self.anchor_direction.split([named[n].numel() for n in self.names])
        for name,piece in zip(self.names,pieces):
            g=piece.reshape_as(named[name]);v=.9*state.get('anchor',{}).get(name,torch.zeros_like(g)).detach()+.1*g;anchor_velocity[name]=v.detach().clone();anchor_steps[name]=.001*v/v.square().mean().clamp_min(.01).sqrt()
        anchor_norm=torch.stack([v.square().sum() for v in anchor_steps.values()]).sum().clamp_min(1e-24).sqrt();anchor_scale=(.2/anchor_norm).clamp_max(1);anchor_steps={n:v*anchor_scale for n,v in anchor_steps.items()}
        radius_norm=torch.stack([v.square().sum() for v in anchor_steps.values()]).sum().clamp_min(1e-6).sqrt();differences={n:named[n].detach()-full[n]-anchor_steps[n] for n in self.names};difference_norm_sq=torch.stack([v.square().sum() for v in differences.values()]).sum();fraction=self.radius*radius_norm/(difference_norm_sq+radius_norm.square()).sqrt();steps={n:anchor_steps[n]+fraction*differences[n] for n in self.names};norm=torch.stack([v.square().sum() for v in steps.values()]).sum().clamp_min(1e-24).sqrt();scale=(.2/norm).clamp_max(1)
        self.last_anchor_step_norm=anchor_norm.detach();self.last_radius=self.radius.detach();self.last_step_deviation=torch.stack([(scale*steps[n]-anchor_steps[n]).square().sum() for n in self.names]).sum().clamp_min(1e-24).sqrt().detach();self.last_trust_bound=(self.radius*radius_norm).detach()
        return {n:p.detach()-scale*steps[n] if n in steps else p.detach().clone() for n,p in named.items()},dict(full=full_velocity,anchor=anchor_velocity)
