"""Receiver-private parameter views backed by the existing packed actor.

Only bank application changes. All packets must contain complete same-version
credit; their generation/aggregation is charged separately.
"""
import torch
from .local_credit_banks import CreditPacket,LocalCreditBanks


class ReceiverCreditBanks:
    def __init__(self,model,**options):
        self.parent_parameters=dict(model.named_parameters());self.mapping={};views=[]
        self.placements={}
        for name,parent in self.parent_parameters.items():
            if name.startswith('core.banks.'):
                field=name.removeprefix('core.banks.')
                for depth in range(parent.shape[0]):
                    for head in range(parent.shape[1]):
                        slots=range(parent.shape[2]) if field!='query' else (None,)
                        for slot in slots:
                            index=(depth,head) if slot is None else (depth,head,slot)
                            bank=f'{name}/d{depth}/h{head}'+('' if slot is None else f'/r{slot}')
                            view=torch.nn.Parameter(parent[index])
                            self.mapping[bank]=(name,index);views.append((bank,view))
                            self.placements[bank]=('head_shared' if slot is None or model.core.tied[field] else 'receiver_private')
            else:
                self.mapping[name]=(name,None);views.append((name,parent));self.placements[name]='shared_interface_or_readout'
        self.local=LocalCreditBanks(views,**options)
        assert sum(v.numel() for _,v in views)==sum(p.numel() for p in self.parent_parameters.values())
    def packets(self,event_id):
        for bank,(name,index) in self.mapping.items():
            gradient=self.parent_parameters[name].grad
            if gradient is not None:
                gradient=gradient if index is None else gradient[index]
                yield CreditPacket(bank,self.local.versions[bank],gradient.detach().clone(),event_id)
    def apply(self,packet):return self.local.apply(packet)
    def storage(self):
        out=self.local.storage()
        out['banks_by_placement']={name:sum(v==name for v in self.placements.values()) for name in set(self.placements.values())}
        return out
