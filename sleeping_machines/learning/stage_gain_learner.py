"""Preserve an acquired learner when migrating fixed residual-gain storage.

Only an optimizer boundary is supported. No depth change or new clock/state
semantics, clipping law, optimizer reset, or fitting occurs here.
"""
import copy
import torch
from ..packed_token_core import PackedTokenCore
from ..stage_gain_token_core import StageGainTokenCore
from .receiver_bank_adamw import ReceiverBankAdamW
from .clone_local_receiver_learner import clone_local_receiver_learner


def migrate_stage_gain_learner(model,optimizer,numeric_state=None):
    if type(model.core) is not PackedTokenCore:
        raise ValueError('Migration requires a legacy packed native core')
    if any(p.grad is not None for p in model.parameters()):
        raise ValueError('Consume and clear pending gradients before learner migration')
    def check(value):
        if isinstance(value,torch.Tensor) and value.requires_grad:
            raise ValueError('Migration requires detached numerical state at an update boundary')
        if isinstance(value,dict):
            for child in value.values():check(child)
        elif isinstance(value,(list,tuple)):
            for child in value:check(child)
    check(numeric_state)
    if type(optimizer) is torch.optim.AdamW:
        owned={id(p) for p in model.parameters()}
        if any(id(p) not in owned for group in optimizer.param_groups for p in group['params']):
            raise ValueError('Optimizer contains parameters from a different actor')
        # One shared deepcopy memo binds all copied optimizer groups to copied
        # model leaves, preserving heterogeneous groups and unused moments.
        cloned,rebuilt=copy.deepcopy((model,optimizer))
        cloned_ids={id(p) for p in cloned.parameters()}
        if any(id(p) not in cloned_ids for group in rebuilt.param_groups for p in group['params']):
            raise ValueError('Optimizer migration lost actor ownership')
    elif type(optimizer) is ReceiverBankAdamW:
        # Rebuild packed-storage views rather than copying aliased Parameters.
        cloned,rebuilt=clone_local_receiver_learner(model,optimizer)
    else:
        raise ValueError('Explicit AdamW or receiver-bank AdamW migration required')
    gains=(float(model.core.gain),)*model.core.depth
    StageGainTokenCore._validate(gains,model.core.depth)
    # Isolated core already owns the cloned optimizer leaves. Change only its
    # fixed configuration representation; replacing leaves would break moments.
    cloned.core.__class__=StageGainTokenCore
    cloned.core.stage_gains=gains
    return cloned,rebuilt,copy.deepcopy(numeric_state)
