"""Native prefix containment for added temporal stages, with separate clock RNG.

This forward primitive keeps the parent's temporal context interface while new
stages are inactive. Activating them changes actual computation, clocks and
feedback. Integrated route/activation learning and bank-optimizer migration are
separate requirements; no training-speed or quality claim is made here.
"""
import copy,inspect
from types import FunctionType
import torch
from .stage_gain_token_core import StageGainTokenCore
from . import sparse_counterfactual_episodes as episodes
from .ordered_token_episodes import ContentFunctions


class OptionalTemporalDepthCore(StageGainTokenCore):
    def get_extra_state(self):
        saved=super().get_extra_state()
        saved.update(parent_depth=self.parent_depth,execution_depth=self.execution_depth,new_depth_rng=self.new_depth_generator.get_state())
        return saved
    def set_extra_state(self,saved):
        super().set_extra_state(saved)
        old=int(saved['parent_depth']);active=int(saved['execution_depth'])
        if not 0<old<=active<=self.depth:raise ValueError('Invalid optional-depth checkpoint')
        self.parent_depth=old;self.execution_depth=active
        self.new_depth_generator=torch.Generator();self.new_depth_generator.set_state(saved['new_depth_rng'])


def grow_native_core(parent, fresh):
    """Preserve every acquired old field; initialize only additional stages."""
    if fresh.depth<=parent.depth:raise ValueError('Added temporal depth required')
    for name in ('sources','payload','heads','pool','content_dim','total_payload'):
        if getattr(parent,name)!=getattr(fresh,name):raise ValueError('Incompatible native interface: '+name)
    grown=StageGainTokenCore.from_packed(parent)
    new=StageGainTokenCore.from_packed(fresh)
    old_depth=parent.depth
    for name,p in grown.banks.items():
        incoming=new.banks[name]
        if p.shape[1:]!=incoming.shape[1:]:raise ValueError('Bank schema changed: '+name)
        grown.banks[name]=torch.nn.Parameter(torch.cat((p.detach(),incoming[old_depth:].detach()),0))
    grown.channel_mix.extend(copy.deepcopy(new.channel_mix[old_depth:]))
    for name in ('transport_rate','transport_frequency'):
        old=getattr(grown,name);extra=getattr(new,name)[old_depth:]
        setattr(grown,name,torch.nn.Parameter(torch.cat((old.detach(),extra.detach()),0)))
    grown.depth=new.depth;grown.stage_gains=tuple(grown.stage_gains)+tuple(new.stage_gains[old_depth:])
    grown.__class__=OptionalTemporalDepthCore
    grown.parent_depth=old_depth;grown.execution_depth=old_depth
    grown.new_depth_generator=torch.Generator().manual_seed(70119)
    return grown


def optional_depth_features(model,tokens,state=None,generator=None,token_content=None,**kwargs):
    """Inactive suffix leaves parent messages/timing/feedback and RNG exact."""
    old_depth=model.parent_depth;active=model.execution_depth
    if not old_depth<=active<=model.depth:raise ValueError('Invalid active temporal depth')
    if kwargs.get('compiled',False):raise ValueError('Containment prototype uses eager native kernels')
    if state is not None and len(state['mem'])<model.depth:
        blank=episodes.initial_state(model,tokens.shape[0]);state=dict(state)
        for field in ('mem','arr','seen'):state[field]=list(state[field])+blank[field][len(state[field]):]
    source=inspect.getsource(episodes.token_features)
    source=source.replace('D, H, U, P = model.depth, model.heads, model.pool, model.payload',
        'D, H, U, P = model.execution_depth, model.heads, model.pool, model.payload')
    # The final active stage owns temporal context, rather than the last
    # allocated stage. Thus dormant stages introduce no delay/transport change.
    source=source.replace('layers = model._stacked(source)','layers = model._stacked(source)[:model.execution_depth]')
    source=source.replace('exponential_(generator=generator) for _ in range(H)',
        'exponential_(generator=generator if depth < model.parent_depth else model.new_depth_generator) for _ in range(H)')
    namespace=dict(episodes.token_features.__globals__)
    if token_content is not None:namespace['F']=ContentFunctions(tokens,token_content)
    exec(compile(source,'<native optional temporal depth>','exec'),namespace)
    return namespace['token_features'](model,tokens,state,generator,**kwargs)
