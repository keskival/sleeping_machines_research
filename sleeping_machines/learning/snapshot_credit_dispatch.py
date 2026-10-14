"""Safe native producer credit delivery through immutable actor storage.

Separate loss backward sweeps are charged. This prototype removes live-actor
read/update aliasing; it is not a sparse replacement for autograd generation.
"""
import copy,time
import torch
from .bank_credit_inbox import BankCreditInbox,CreditFragment

class SnapshotCreditDispatch:
    def __init__(self,model,receiver,event_id,producers):
        self.receiver=receiver;self.producers=tuple(producers)
        if not self.producers or len(set(self.producers))!=len(self.producers):raise ValueError('Unique producers required')
        began=time.monotonic();self.actor=copy.deepcopy(model);self.snapshot_s=time.monotonic()-began
        self.snapshot_parameters=dict(self.actor.named_parameters())
        if set(self.snapshot_parameters)!=set(receiver.parent_parameters):raise ValueError('Snapshot schema changed')
        for name,p in self.snapshot_parameters.items():
            live=receiver.parent_parameters[name]
            if p.shape!=live.shape or p.data_ptr()==live.data_ptr():raise ValueError('Snapshot aliases live parameters')
        self.box=BankCreditInbox(receiver.local);self.event_id=event_id;self.active=None
        self.completed=[];self.delivered=set();self.used=set();self.handles=[];self.peak_fragment_bytes=0;self.applied=[];self.backward_s=0.;self.in_autograd=False;self.applied_during_backward=0
        self.by_parent={name:[] for name in self.snapshot_parameters}
        for bank,(name,index) in receiver.mapping.items():
            self.by_parent[name].append((bank,index));self.box.issue(bank,receiver.local.versions[bank],event_id,self.producers)
        for name,p in self.snapshot_parameters.items():
            if p.requires_grad:self.handles.append(p.register_hook(self._hook(name)))
    def _receive(self,bank,g,used):
        if bank in self.delivered:raise ValueError('Duplicate parent gradient completion')
        self.delivered.add(bank)
        if used:self.used.add(bank)
        ready=self.box.receive(CreditFragment(bank,self.receiver.local.versions[bank],self.active,self.event_id,g))
        self.peak_fragment_bytes=max(self.peak_fragment_bytes,self.box.storage()['retained_gradient_bytes'])
        if ready:
            packet=self.box.take(bank)
            # Match AdamW's None-gradient skip, including weight decay/moments.
            if bank in self.used:
                self.receiver.apply(packet);self.applied.append(bank)
                self.applied_during_backward+=int(self.in_autograd)
    def _hook(self,name):
        def receive(g):
            if self.active is None:raise ValueError('Backward outside declared producer')
            for bank,index in self.by_parent[name]:self._receive(bank,g if index is None else g[index],True)
            return g
        return receive
    def backward(self,producer,loss,retain_graph=False):
        if self.active is not None or producer not in self.producers or producer in self.completed:raise ValueError('Invalid producer')
        # Canonical producer order also preserves None-gradient handling.
        if producer!=self.producers[len(self.completed)]:raise ValueError('Use declared producer order')
        self.active=producer;self.delivered=set();self.actor.zero_grad(set_to_none=True);began=time.monotonic()
        try:
            self.in_autograd=True
            loss.backward(retain_graph=retain_graph)
            self.in_autograd=False
            for bank,p in self.receiver.local.parameters.items():
                if bank not in self.delivered:self._receive(bank,torch.zeros_like(p),False)
            self.completed.append(producer)
        finally:self.backward_s+=time.monotonic()-began;self.active=None;self.in_autograd=False
    def finish(self):
        if tuple(self.completed)!=self.producers or self.box.pending:raise ValueError('Incomplete producer event')
        for h in self.handles:h.remove()
        self.handles=[]
        return dict(snapshot_s=self.snapshot_s,backward_s=self.backward_s,snapshot_parameter_bytes=sum(p.numel()*p.element_size() for p in self.snapshot_parameters.values()),peak_fragment_bytes=self.peak_fragment_bytes,applied_banks=len(self.applied),applied_during_backward=self.applied_during_backward,producers=len(self.producers),scope='Immutable actor reads; live bank updates during snapshot backward. Separate producer sweeps and full snapshot paid; native event-local gradient kernel not implemented.')
