"""B4 development: separate mark-branch inputs from the physical-time core."""
import copy
import torch
from race_tpp_logspread import RaceTPP as Stable
from race_tpp_logspread_marks import RaceTPP as MarkModel


class RaceTPP(MarkModel):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.mark_input_embed=copy.deepcopy(self.embed)
        self.mark_input_gap=copy.deepcopy(self.gap)

    def encode(self,t,m,mask):
        h,slots=Stable.encode(self,t,m,mask)
        dt=torch.diff(t,dim=1,prepend=t[:,:1]).clamp_min(0)/self.scale
        age=(t-t[:,:1])/self.scale
        y=self.mark_input_embed(m)+self.mark_input_gap(torch.stack([torch.log1p(dt),(dt>0).to(t.dtype),torch.log1p(age)],-1))
        for layer in self.mark_layers:y=layer(y,torch.ones_like(dt))
        self._extra=self.mark_extra(y)
        return h,slots
