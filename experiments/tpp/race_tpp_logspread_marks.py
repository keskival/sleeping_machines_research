"""B4 development: repaired clocks plus established per-event mark memory.

Timing substrate is unchanged. This auxiliary persistent branch reads event
history without elapsed-time erasure and enters mark distributions only.
Core parameter initialization stays identical to the log-spread model.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from race_tpp_logspread import RaceTPP as Stable
from race_tpp_v20 import TemporalMemoryLayer, RaceTPP as MarkReference


class RaceTPP(Stable):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        d=self.embed.embedding_dim; modes=self.layers[0].n; dropout=self.layers[0].drop.p
        self.mark_mem_n=1; self.mark_stats=False
        self.mark_layers=nn.ModuleList([TemporalMemoryLayer(d,modes,dropout)])
        self.mark_extra=nn.Linear(d,self.K)
        nn.init.zeros_(self.mark_extra.weight);nn.init.zeros_(self.mark_extra.bias)
        self._extra=None

    encode=MarkReference.encode
    state_eval=MarkReference.state_eval

    def clocks(self,h,slots):
        rate,mu,sigma,weight,logpk=super().clocks(h,slots)
        if self._extra is not None:
            logpk=F.log_softmax(logpk+self._extra[:,:h.shape[1]].unsqueeze(-2),-1)
        return rate,mu,sigma,weight,logpk
