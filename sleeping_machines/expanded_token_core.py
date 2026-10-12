"""Grow addressed banks without changing existing optimizer leaves.

This storage adapter adds competing receivers; it does not preserve the old
route distribution or make zero-cost discovery claims. Old and new receiver
leaves retain independent Adam ages through optimizer_for_expanded_model.
"""
import copy
import torch
from torch import nn
from torch.nn import functional as F
from .packed_token_core import PackedTokenCore, UNIT_FIELDS


class ExpandedTokenCore(PackedTokenCore):
    def __init__(self, source, additional_receivers):
        nn.Module.__init__(self)
        if not isinstance(source, PackedTokenCore) or isinstance(source, ExpandedTokenCore):
            raise TypeError('Expand an original PackedTokenCore; recursive expansion needs an explicit bank map')
        if isinstance(additional_receivers, bool) or not isinstance(additional_receivers, int) or additional_receivers < 1:
            raise ValueError('Positive integer additional receivers required')
        for name in ('sources','content_dim','classes','payload','depth','heads','total_payload','credit','gain'):
            setattr(self,name,copy.deepcopy(getattr(source,name)))
        self.pool=source.pool+additional_receivers
        self.original_pool=source.pool
        self.tied=copy.deepcopy(source.tied)
        for name,module in source.named_children():
            self.add_module(name,copy.deepcopy(module))
        for name,parameter in source.named_parameters(recurse=False):
            self.register_parameter(name,nn.Parameter(parameter.detach().clone(),requires_grad=parameter.requires_grad))
        self.extension_banks=nn.ParameterDict()
        # Explicit cyclic cloning is an initialization, not inferred route
        # similarity. Each added receiver has independent persistent state.
        index=torch.arange(additional_receivers,device=source.banks['key'].device)%source.pool
        for name in UNIT_FIELDS:
            if not self.tied[name]:
                old=source.banks[name]
                self.extension_banks[name]=nn.Parameter(old.detach().index_select(2,index).clone(),requires_grad=old.requires_grad)
        self.train(source.training)

    def _stacked(self,source):
        if source != 0:raise ValueError('One observed token stream source required')
        layers=[]
        for depth in range(self.depth):
            bank={}
            for name in UNIT_FIELDS:
                value=self.banks[name][depth]
                if self.tied[name]:
                    value=value.expand(self.heads,self.pool,*value.shape[2:])
                else:
                    value=torch.cat((value,self.extension_banks[name][depth]),dim=1)
                bank[name]=value.reshape(self.heads*self.pool,*value.shape[2:])
            bank['rate']=F.softplus(bank.pop('raw_rate'))+1e-6
            bank['frequency']=bank['frequency'].to(torch.float64)
            bank['query']=self.banks['query'][depth]
            bank['gain']=self.gain
            layers.append(bank)
        return layers

    def expand_numeric_state(self,state):
        """Copy retained physical state; new receivers start unseen and empty."""
        if state is None:return None
        result={}
        for name,value in state.items():
            if name in ('mem','arr','seen'):
                if len(value)!=self.depth:raise ValueError('State depth differs')
                result[name]=[]
                for old in value:
                    if old.shape[1:3]!=(self.heads,self.original_pool):raise ValueError('State receiver shape differs')
                    shape=list(old.shape);shape[2]=self.pool-self.original_pool
                    result[name].append(torch.cat((old.detach().clone(),old.new_zeros(shape)),dim=2))
            elif name=='last_writes':
                if tuple(value.shape)!=(self.depth,self.heads,self.original_pool):raise ValueError('Write counter shape differs')
                shape=list(value.shape);shape[2]=self.pool-self.original_pool
                result[name]=torch.cat((value.detach().clone(),value.new_zeros(shape)),dim=2)
            elif isinstance(value,torch.Tensor):result[name]=value.detach().clone()
            else:raise ValueError('Unknown non-tensor numerical state')
        return result

    def original_gradient(self,name):
        parts=name.split('.')
        if parts[0]=='units':
            depth,head,source,receiver=map(int,parts[1:5])
            if source!=0 or not 0<=receiver<self.pool:raise ValueError('Invalid receiver address')
            path=tuple(parts[5:]);field=next(k for k,v in UNIT_FIELDS.items() if v==path)
            if self.tied[field]:return self.banks[field].grad[depth,head,0]
            if receiver>=self.original_pool:
                return self.extension_banks[field].grad[depth,head,receiver-self.original_pool]
        return super().original_gradient(name)
