"""Rate-independent multiscale evidence fusion with direct per-level credit.

Feature gates combine potentially complementary evidence, not probabilities.
Single-level heads are auxiliary diagnostics/learning paths, not assumed capable
of solving tasks whose target requires cross-level interactions.
"""
import torch
from torch import nn
from torch.nn import functional as F


class ResponsibilityReadout(nn.Module):
    def __init__(self,levels,width,classes,initial_slope=1.):
        super().__init__();self.levels=levels;self.width=width
        self.maps=nn.ModuleList(nn.Linear(width,width,bias=False) for _ in range(levels))
        for m in self.maps:nn.init.eye_(m.weight)
        self.head=nn.Linear(width,classes)
        self.gate=nn.Parameter(-initial_slope*torch.arange(levels,dtype=torch.float32))
        self.query=nn.Linear(width,levels,bias=False);nn.init.zeros_(self.query.weight)

    def forward(self,views,available=None):
        if views.shape!=(self.levels,self.width):raise ValueError('one typed persistent view per level')
        if available is None:available=torch.ones(self.levels,dtype=torch.bool,device=views.device)
        if available.shape!=(self.levels,) or available.dtype!=torch.bool or not bool(available.any()):
            raise ValueError('nonempty boolean availability required')
        values=torch.stack([m(v) for m,v in zip(self.maps,views)])
        # Query from the latest available abstraction; no event-count term.
        latest=int(torch.nonzero(available)[-1,0])
        scores=self.gate+self.query(values[latest]);weights=scores.masked_fill(~available,-torch.inf).softmax(0)
        fused=(weights[:,None]*values).sum(0)
        return self.head(fused),dict(weights=weights,values=values,single_logits=self.head(values),available=available,
                                      allocation_level_expectation=(weights*torch.arange(self.levels,device=views.device)).sum())

    def objective(self,logits,record,target,auxiliary_weight=.1):
        target=torch.as_tensor(target,dtype=torch.long,device=logits.device).reshape(1)
        fused=F.cross_entropy(logits[None],target)
        available=record['available'];single=record['single_logits'][available]
        auxiliary=F.cross_entropy(single,target.expand(len(single)))
        return fused+auxiliary_weight*auxiliary,dict(prediction=fused,auxiliary=auxiliary)

    def gate_value_effect(self,record):
        # Exact frozen leave-one-source-out prediction, keeping original
        # normalization: a source cut, not a change in the other source weights.
        contribution=record['weights'][:,None]*record['values']
        total=contribution.sum(0)
        return torch.stack([self.head(total-c) for c in contribution])
