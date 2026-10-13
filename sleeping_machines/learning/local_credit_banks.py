"""Versioned local credit packets and independent per-leaf Adam banks.

Credit generation is external: this does not remove history backpropagation.
A packet consumes one local update/version; stale packets are rejected. Local
clipping is a separate learning rule from model-wide clipping.
"""
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class CreditPacket:
    bank: str
    version: int
    gradient: torch.Tensor
    event_id: str


class LocalCreditBanks:
    def __init__(self,named_parameters,lr=.003,local_clip=None,**adam_options):
        if local_clip is not None and local_clip<=0:raise ValueError('Positive local clip required')
        self.parameters=dict(named_parameters)
        if not self.parameters:raise ValueError('At least one parameter bank required')
        if len({id(p) for p in self.parameters.values()})!=len(self.parameters):raise ValueError('Shared leaves require one canonical bank')
        self.optimizers={name:torch.optim.AdamW([p],lr=lr,**adam_options) for name,p in self.parameters.items()}
        self.versions={name:0 for name in self.parameters};self.local_clip=local_clip
        self.receipts=[]
    def apply(self,packet):
        if packet.bank not in self.parameters:raise ValueError('Unknown bank')
        if packet.version!=self.versions[packet.bank]:raise ValueError('Stale or future credit packet')
        p=self.parameters[packet.bank];g=packet.gradient
        if g.shape!=p.shape or g.dtype!=p.dtype or g.device!=p.device:raise ValueError('Gradient schema mismatch')
        if not bool(torch.isfinite(g).all()):raise ValueError('Nonfinite credit')
        g=g.detach().clone()
        norm=float(g.norm())
        if self.local_clip is not None:g.mul_(min(1.,self.local_clip/(norm+1e-6)))
        p.grad=g;optimizer=self.optimizers[packet.bank];optimizer.step();optimizer.zero_grad(set_to_none=True)
        self.versions[packet.bank]+=1
        receipt=dict(bank=packet.bank,issued_version=packet.version,applied_version=self.versions[packet.bank],
            event_id=packet.event_id,gradient_bytes=g.numel()*g.element_size(),local_clip=self.local_clip)
        self.receipts.append(receipt);return receipt
    def storage(self):
        return dict(parameter_bytes=sum(p.numel()*p.element_size() for p in self.parameters.values()),
            optimizer_bytes=sum(v.numel()*v.element_size() for opt in self.optimizers.values() for s in opt.state.values() for v in s.values() if isinstance(v,torch.Tensor)),
            local_version_counters=len(self.versions))
