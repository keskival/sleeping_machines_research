"""Clone the actual local-clipped learner including packed storage and moments.

Deep-copying a tuple of packed actor and Parameter views alone does not promise
that the cloned views still address the cloned packed actor's storage.
"""
import copy
from .receiver_bank_adamw import ReceiverBankAdamW


def clone_local_receiver_learner(model,optimizer):
    if optimizer.model is not model:
        raise ValueError('Optimizer belongs to another actor')
    allowed=('lr','betas','eps','weight_decay','amsgrad','foreach','maximize','capturable','differentiable','fused')
    groups=[o.param_groups[0] for o in optimizer.receiver.local.optimizers.values()]
    options={k:groups[0][k] for k in allowed if k in groups[0]}
    if any({k:g[k] for k in allowed if k in g}!=options for g in groups):
        raise ValueError('Heterogeneous bank options need an explicit clone recipe')
    if hasattr(optimizer.receiver.local,'bank_caps'):
        raise ValueError('Calibrated bank-cap learner requires an explicit cap-preserving clone recipe')
    cloned=copy.deepcopy(model)
    rebuilt=ReceiverBankAdamW(cloned,local_clip=optimizer.receiver.local.local_clip,**options)
    rebuilt.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    if rebuilt.receiver.local.local_clip!=optimizer.receiver.local.local_clip:
        raise ValueError('Clone changed local clipping law')
    if rebuilt.receiver.local.versions!=optimizer.receiver.local.versions:
        raise ValueError('Clone changed issued bank versions')
    for bank,(name,index) in rebuilt.receiver.mapping.items():
        view=rebuilt.receiver.local.parameters[bank]
        parent=rebuilt.receiver.parent_parameters[name]
        if view.untyped_storage().data_ptr()!=parent.untyped_storage().data_ptr():
            raise ValueError('Cloned optimizer view lost its backing actor bank')
        if parent.untyped_storage().data_ptr()==optimizer.receiver.parent_parameters[name].untyped_storage().data_ptr():
            raise ValueError('Cloned actor shares mutable source storage')
    return cloned,rebuilt
