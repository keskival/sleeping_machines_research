"""Parameter-bank representation of the existing integrated addressed core.

Changes storage/optimizer dispatch, not routes, clocks or memory computation.
Untied matrices are viewed directly; tied matrices expand across receiver slots.
"""
import copy
import torch
from torch import nn
from torch.nn import functional as F


UNIT_FIELDS = {
    'key': ('key',), 'key_read': ('key_read','weight'), 'clock_bias': ('clock_bias',),
    'input': ('input','weight'), 'output': ('output','weight'),
    'gate_w': ('gate','weight'), 'gate_b': ('gate','bias'),
    'control_w': ('control','weight'), 'control_b': ('control','bias'),
    'raw_rate': ('raw_rate',), 'frequency': ('frequency',),
}


def unit_field(unit,path):
    value=unit
    for name in path:value=getattr(value,name)
    return value


class PackedTokenCore(nn.Module):
    def __init__(self,original):
        super().__init__()
        if original.sources != 1:
            raise ValueError('This token-core adapter packs the one-source language configuration')
        for name in ('sources','content_dim','classes','payload','depth','pool','heads','total_payload','credit'):
            setattr(self,name,getattr(original,name))
        for name in ('embedding','content','source_gate','channel_mix','head'):
            setattr(self,name,copy.deepcopy(getattr(original,name)))
        self.transport_rate=nn.Parameter(original.transport_rate.detach().clone())
        self.transport_frequency=nn.Parameter(original.transport_frequency.detach().clone())
        self.gain=original.units[0][0][0][0].gain
        self.banks=nn.ParameterDict()
        self.tied={}
        for name,path in UNIT_FIELDS.items():
            sharing=[]
            for layer in original.units:
                for head in layer:
                    pool=head[0]
                    sharing.append(len({id(unit_field(u,path)) for u in pool})==1)
            if any(sharing) and not all(sharing):
                raise ValueError('Mixed per-head sharing requires a separate packing map')
            tied=all(sharing);self.tied[name]=tied
            values=[]
            for layer in original.units:
                heads=[]
                for head in layer:
                    pool=head[0]
                    if any(u.gain!=self.gain for u in pool):
                        raise ValueError('This adapter expects the existing uniform unit gain')
                    heads.append(torch.stack([unit_field(u,path).detach() for u in (pool[:1] if tied else pool)]))
                values.append(torch.stack(heads))
            self.banks[name]=nn.Parameter(torch.stack(values))
        self.banks['query']=nn.Parameter(torch.stack([torch.stack([q.weight.detach() for q in layer])
                                                    for layer in original.queries]))

    def _stacked(self,source):
        if source != 0:raise ValueError('One observed token stream source required')
        layers=[]
        for depth in range(self.depth):
            bank={}
            for name in UNIT_FIELDS:
                value=self.banks[name][depth]
                if self.tied[name]:value=value.expand(self.heads,self.pool,*value.shape[2:])
                bank[name]=value.reshape(self.heads*self.pool,*value.shape[2:])
            bank['rate']=F.softplus(bank.pop('raw_rate'))+1e-6
            bank['frequency']=bank['frequency'].to(torch.float64)
            bank['query']=self.banks['query'][depth]
            bank['gain']=self.gain
            layers.append(bank)
        return layers

    def original_gradient(self,name):
        """Map an original named leaf to its bank gradient for parity contracts."""
        parts=name.split('.')
        if parts[0]=='queries':
            return self.banks['query'].grad[int(parts[1]),int(parts[2])]
        if parts[0]=='units':
            depth,head,source,receiver=map(int,parts[1:5])
            if source != 0:raise ValueError('One-source gradient map')
            path=tuple(parts[5:]);field=next(k for k,v in UNIT_FIELDS.items() if v==path)
            return self.banks[field].grad[depth,head,0 if self.tied[field] else receiver]
        return dict(self.named_parameters())[name].grad
