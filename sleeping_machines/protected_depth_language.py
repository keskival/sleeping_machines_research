"""Strong acquired token model plus stateful native temporal feature correction.

The parent's readout coordinates and feedback remain protected. Correction
features are produced by genuine added native stages with private clock/state.
At zero correction output weights the old prediction and factual credit agree.
Internal correction-stage credit starts as the output projection learns.
"""
import copy
import torch
from torch import nn
from .optional_temporal_depth import grow_native_core
from .protected_temporal_branch import protected_branch_features
from .ordered_token_lookup import ordered_content
from .sparse_counterfactual_episodes import detach
token_features=protected_branch_features

class TemporalCorrectionReadout(nn.Module):
    def __init__(self,base,parent_width,branch_width):
        super().__init__();self.base=base;self.parent_width=parent_width
        with torch.random.fork_rng():self.correction=nn.Linear(branch_width,parent_width,bias=False,device=next(base.parameters()).device,dtype=next(base.parameters()).dtype)
        nn.init.zeros_(self.correction.weight)
    def features(self,values):return values[...,:self.parent_width]+self.correction(values[...,self.parent_width:])
    def nll(self,values,targets):return self.base.nll(self.features(values),targets)
    def __getattr__(self,name):
        try:return super().__getattr__(name)
        except AttributeError:return getattr(super().__getattr__('base'),name)

class ProtectedDepthForward:
    def forward(self,ids,state,generator,args,deterministic=False):
        features=[];probabilities=[];winners=[];writes=None;content=ordered_content(self.core.content.weight,ids)
        if not self.shadow_mode:self.replay_windows={}
        for begin in range(0,ids.shape[1],args.credit_window):
            end=min(begin+args.credit_window,ids.shape[1]);force=mask=None
            if self.force_site is not None and begin<=self.force_site[0]<end:force=(self.force_site[0]-begin,*self.force_site[1:])
            if self.suppress_site is not None and begin<=self.suppress_site[0]<end:mask=(self.suppress_site[0]-begin,*self.suppress_site[1:])
            if not self.shadow_mode:
                window=dict(begin=begin,state=clone_state(state),rng=generator.get_state().clone(),new_depth_rng=self.core.new_depth_generator.get_state().clone())
                self.replay_windows[begin]=window
                if mask is not None:self.replay_window=window
            x,state,pis=token_features(self.core,ids[:,begin:end],state,generator,
                token_content=content[begin:end],route_credit='none',deterministic=deterministic,
                eos=50256,free_bias=args.free_bias,temperature=1. if deterministic else args.temperature,
                force_site=force,suppress_site=mask,alternative_generator=self.local_generator,compiled=False,
                parent_readout='all',pair=True,output='parent')
            features.append(x);probabilities.extend(pis);winners.append(state['race_winners'])
            writes=state['last_writes'] if writes is None else writes+state['last_writes']
            if end<ids.shape[1]:state=detach(state)
        state['race_winners']=torch.cat(winners,0);state['last_writes']=writes
        return torch.cat(features,1),state,probabilities


def clone_state(state):
    if state is None:return None
    return {k:[t.detach().clone() for t in v] if isinstance(v,list) else v.detach().clone() for k,v in state.items()}

def grow_protected_language(parent,fresh):
    """Retain the trained all-parent-depth readout, initialize only new blocks."""
    grown=copy.deepcopy(parent);grown.core=grow_native_core(parent.core,fresh.core)
    # Copy a compatible JOINT trained temporal operator prior, rather than
    # drawing independent marginal histograms for each parameter field.
    old=parent.core.depth
    with torch.no_grad():
        for d in range(old,grown.core.depth):
            for name,value in grown.core.banks.items():value[d].copy_(parent.core.banks[name][-1])
            grown.core.channel_mix[d].load_state_dict(parent.core.channel_mix[-1].state_dict())
            grown.core.transport_rate[d].copy_(parent.core.transport_rate[-1]);grown.core.transport_frequency[d].copy_(parent.core.transport_frequency[-1])
    grown.readout=TemporalCorrectionReadout(grown.readout,parent.core.depth*parent.core.total_payload,parent.core.total_payload)
    grown.__class__=type('Protected'+parent.__class__.__name__,(ProtectedDepthForward,parent.__class__),{})
    return grown
