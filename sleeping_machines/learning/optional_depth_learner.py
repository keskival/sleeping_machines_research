"""Preserve acquired receiver Adam moments/versions across optional native depth.

New caps must be supplied explicitly for calibrated learners. This does not
calibrate those caps, select activation, or fit a growth model.
"""
import copy,inspect
import torch
from .receiver_bank_adamw import ReceiverBankAdamW
from .calibrated_receiver_adamw import CalibratedReceiverBankAdamW
from ..optional_temporal_depth import grow_native_core

def migrate_optional_depth_learner(parent,optimizer,fresh_core,numeric_state=None,new_bank_caps=None):
    if type(optimizer) not in (ReceiverBankAdamW,CalibratedReceiverBankAdamW):raise ValueError('Explicit receiver bank Adam policy required')
    if optimizer.model is not parent:raise ValueError('Optimizer actor ownership mismatch')
    if any(p.grad is not None for p in parent.parameters()):raise ValueError('Clear pending credit before growth')
    saved=optimizer.state_dict();grown=copy.deepcopy(parent);grown.core=grow_native_core(parent.core,fresh_core)
    first=next(iter(optimizer.receiver.local.optimizers.values())).param_groups[0]
    allowed=set(inspect.signature(torch.optim.AdamW).parameters)-{'params'}
    options={k:copy.deepcopy(v) for k,v in first.items() if k in allowed}
    if type(optimizer) is CalibratedReceiverBankAdamW:
        if new_bank_caps is None:raise ValueError('TRAIN-calibrated complete grown bank caps required')
        old_caps=optimizer.receiver.local.bank_caps
        if any(new_bank_caps.get(b)!=c for b,c in old_caps.items()):raise ValueError('Neutral migration must preserve old caps; recalibration is a separate intervention')
        rebuilt=CalibratedReceiverBankAdamW(grown,new_bank_caps,**options)
    else:rebuilt=ReceiverBankAdamW(grown,**options)
    for bank,old_opt in optimizer.receiver.local.optimizers.items():
        if bank not in rebuilt.receiver.local.optimizers:raise ValueError('Lost old bank: '+bank)
        new_opt=rebuilt.receiver.local.optimizers[bank];state=copy.deepcopy(old_opt.state_dict())
        old_param=optimizer.receiver.local.parameters[bank];new_param=rebuilt.receiver.local.parameters[bank]
        for value in state['state'].values():
            for key,t in list(value.items()):
                if torch.is_tensor(t) and t.shape==old_param.shape and t.shape!=new_param.shape:
                    if t.ndim!=new_param.ndim or t.shape[1:]!=new_param.shape[1:] or t.shape[0]>new_param.shape[0]:raise ValueError('Unsupported moment shape migration')
                    padded=torch.zeros_like(new_param);padded[:t.shape[0]].copy_(t);value[key]=padded
        new_opt.load_state_dict(state)
        rebuilt.receiver.local.versions[bank]=optimizer.receiver.local.versions[bank]
    rebuilt.update=optimizer.update;rebuilt.deliveries=optimizer.deliveries
    # New banks begin with version0 and empty Adam state; old shared transport
    # arrays keep their old Adam step, with zero moments in added coordinates.
    return grown,rebuilt,copy.deepcopy(numeric_state)
