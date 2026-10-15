"""Growth preserves old local learning units, including transport-prefix ownership.

Appending new depths must not put their gradients into an acquired bank's norm
or Adam step. Old banks retain their caps, moments and versions exactly. New
banks own their views and require separately calibrated caps before fitting.
"""
import copy
import torch
from .receiver_bank_adamw import ReceiverBankAdamW
from .receiver_credit_banks import ReceiverCreditBanks
from .local_credit_banks import LocalCreditBanks
from .calibrated_receiver_adamw import CalibratedLocalCreditBanks

class ProtectedGrowthAdamW(ReceiverBankAdamW):
    def __init__(self,model,parent_depth,bank_caps=None,lr=.003,weight_decay=.01,**options):
        self.model=model;self.receiver=ReceiverCreditBanks(model,lr=lr,weight_decay=weight_decay,**options);self.update=0;self.deliveries=0
        mapping=self.receiver.mapping;placements=self.receiver.placements;views=dict(self.receiver.local.parameters)
        for name in ('core.transport_rate','core.transport_frequency'):
            p=self.receiver.parent_parameters[name]
            if not 0<parent_depth<p.shape[0]:raise ValueError('Declare acquired depth prefix')
            mapping[name]=(name,(slice(0,parent_depth),));views[name]=torch.nn.Parameter(p[:parent_depth])
            for d in range(parent_depth,p.shape[0]):
                bank=f'{name}/d{d}';mapping[bank]=(name,(d,));views[bank]=torch.nn.Parameter(p[d]);placements[bank]='new_depth_transport'
        constructor=LocalCreditBanks if bank_caps is None else CalibratedLocalCreditBanks
        args={} if bank_caps is None else {'bank_caps':bank_caps}
        self.receiver.local=constructor(views.items(),lr=lr,weight_decay=weight_decay,**args,**options)
        assert sum(p.numel() for p in views.values())==sum(p.numel() for p in model.parameters())
    def state_dict(self):
        value=super().state_dict()
        if hasattr(self.receiver.local,'bank_caps'):value['bank_caps']=dict(self.receiver.local.bank_caps)
        return value
    def load_state_dict(self,value):
        if value.get('bank_caps')!=getattr(self.receiver.local,'bank_caps',None):raise ValueError('Changed growth caps')
        super().load_state_dict(value)

def migrate_parent_banks(parent_optimizer,grown,new_caps):
    old=parent_optimizer.receiver.local
    if not hasattr(old,'bank_caps'):raise ValueError('Acquired calibrated policy required')
    options=copy.deepcopy(next(iter(old.optimizers.values())).param_groups[0]);options.pop('params')
    import inspect
    options={k:v for k,v in options.items() if k in inspect.signature(torch.optim.AdamW).parameters}
    probe=ProtectedGrowthAdamW(grown,grown.core.parent_depth,**options);caps={};correspondence={}
    for bank,p in old.parameters.items():
        target=bank.replace('readout.','readout.base.',1) if bank.startswith('readout.') else bank
        if target not in probe.receiver.local.parameters or p.shape!=probe.receiver.local.parameters[target].shape:raise ValueError('Acquired ownership changed: '+bank)
        caps[target]=old.bank_caps[bank];correspondence[bank]=target
    missing=set(probe.receiver.mapping)-set(caps)
    if set(new_caps)!=missing:raise ValueError('Supply caps for exactly the new banks')
    caps.update(new_caps);result=ProtectedGrowthAdamW(grown,grown.core.parent_depth,caps,**options)
    for bank,target in correspondence.items():
        result.receiver.local.optimizers[target].load_state_dict(copy.deepcopy(old.optimizers[bank].state_dict()));result.receiver.local.versions[target]=old.versions[bank]
    result.update=parent_optimizer.update;result.deliveries=parent_optimizer.deliveries
    return result,correspondence
