"""Map acquired dense Adam moments to receiver views, preserving named ownership."""
import copy,inspect
import torch
from .receiver_bank_adamw import ReceiverBankAdamW

def acquired_receiver_optimizer(parent,dense_optimizer,candidate):
    if type(dense_optimizer) is not torch.optim.AdamW:raise ValueError('AdamW source required')
    old=dict(parent.named_parameters());group=dense_optimizer.param_groups[0]
    allowed=set(inspect.signature(torch.optim.AdamW).parameters)-{'params'}
    options={k:copy.deepcopy(v) for k,v in group.items() if k in allowed}
    result=ReceiverBankAdamW(candidate,**options)
    for bank,(name,index) in result.receiver.mapping.items():
        original_name=name.replace('readout.base.','readout.',1) if name.startswith('readout.base.') else name
        if original_name not in old:continue
        previous=old[original_name];target=result.receiver.local.parameters[bank]
        if index is not None and index[0]>=previous.shape[0]:continue
        state=dense_optimizer.state.get(previous,{})
        migrated={}
        for key,value in state.items():
            if not torch.is_tensor(value):migrated[key]=copy.deepcopy(value);continue
            v=value.detach().clone()
            if v.shape==previous.shape:
                if index is not None:v=v[index].clone()
                elif v.shape!=target.shape:
                    if v.ndim!=target.ndim or v.shape[1:]!=target.shape[1:] or v.shape[0]>target.shape[0]:raise ValueError('Moment shape changed beyond depth')
                    padded=torch.zeros_like(target);padded[:v.shape[0]].copy_(v);v=padded
            migrated[key]=v
        if migrated:result.receiver.local.optimizers[bank].state[target]=migrated
    return result
