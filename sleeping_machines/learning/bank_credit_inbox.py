"""Bank-local readiness for complete same-version credit fragments.

This removes a readiness barrier, not gradient-generation work. Caller supplies
causal producer completion sets and prevents mixed-version actor reads.
"""
from dataclasses import dataclass
import torch
from .local_credit_banks import CreditPacket

@dataclass(frozen=True)
class CreditFragment:
    bank: str
    version: int
    producer: str
    event_id: str
    gradient: torch.Tensor

class BankCreditInbox:
    def __init__(self,banks):
        self.banks=banks;self.pending={};self.fragments_received=0
    def issue(self,bank,version,event_id,producers):
        if bank not in self.banks.parameters or version!=self.banks.versions[bank]:raise ValueError('Unknown bank or stale issue')
        if bank in self.pending:raise ValueError('Bank already has an outstanding credit event')
        order=tuple(producers)
        if not order or len(set(order))!=len(order) or not all(isinstance(p,str) and p for p in order):raise ValueError('Distinct declared producers required')
        if not event_id:raise ValueError('Event identity required')
        self.pending[bank]=dict(version=version,event_id=event_id,order=order,values={})
    def receive(self,fragment):
        if fragment.bank not in self.pending:raise ValueError('No issued event for bank')
        entry=self.pending[fragment.bank];p=self.banks.parameters[fragment.bank];g=fragment.gradient
        if fragment.version!=entry['version'] or fragment.version!=self.banks.versions[fragment.bank]:raise ValueError('Stale credit fragment')
        if fragment.event_id!=entry['event_id'] or fragment.producer not in entry['order']:raise ValueError('Wrong event or undeclared producer')
        if fragment.producer in entry['values']:raise ValueError('Duplicate contribution')
        if g.shape!=p.shape or g.dtype!=p.dtype or g.device!=p.device or not bool(torch.isfinite(g).all()):raise ValueError('Invalid fragment schema/value')
        entry['values'][fragment.producer]=g.detach().clone();self.fragments_received+=1
        return len(entry['values'])==len(entry['order'])
    def ready(self,bank):
        e=self.pending[bank];return len(e['values'])==len(e['order'])
    def take(self,bank):
        e=self.pending[bank]
        if not self.ready(bank):raise ValueError('Bank credit is incomplete')
        if self.banks.versions[bank]!=e['version']:raise ValueError('Bank changed before credit delivery')
        # Declared canonical order preserves arithmetic under arbitrary arrival.
        total=e['values'][e['order'][0]].clone()
        for producer in e['order'][1:]:total.add_(e['values'][producer])
        if not bool(torch.isfinite(total).all()):raise ValueError('Aggregate credit overflow')
        packet=CreditPacket(bank,e['version'],total,e['event_id']);del self.pending[bank]
        return packet
    def storage(self):
        return dict(outstanding_banks=len(self.pending),retained_gradient_bytes=sum(g.numel()*g.element_size() for e in self.pending.values() for g in e['values'].values()),producer_metadata_entries=sum(len(e['order']) for e in self.pending.values()))
