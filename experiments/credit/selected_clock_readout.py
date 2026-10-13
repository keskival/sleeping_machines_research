"""Exact selected-row implementation of fixed-support native clock readout."""
import copy
import torch
from native_addressed_clock_credit import AddressedClockCredit,support

class SelectedLinear(torch.nn.Linear):
    def __init__(self,inputs):
        super().__init__(inputs,64);self.indices=torch.tensor([0,16,32,48]);self.calls=0
    def forward(self,x):
        self.calls+=1;indices=self.indices.to(x.device)
        values=torch.nn.functional.linear(x,self.weight.index_select(0,indices),self.bias.index_select(0,indices))
        return x.new_zeros(*x.shape[:-1],64).index_copy(-1,indices,values)

class SelectedClockCredit(AddressedClockCredit):
    def __init__(self,parent,scale):
        super().__init__(parent,scale)
        with torch.random.fork_rng(devices=[]):head=SelectedLinear(self.core.total_payload).double()
        head.load_state_dict(self.core.head.state_dict());self.core.head=head
    def forward(self,features,observed,carry,operator):
        version=0 if carry is None or carry.parent is None else carry.parent.version
        self.core.head.indices=support(version).nonzero().flatten()
        return super().forward(features,observed,carry,operator)

def compile_readout(critic):
    with torch.random.fork_rng(devices=[]):compiled=SelectedClockCredit(critic.parent,critic.scale)
    compiled.load_state_dict(critic.state_dict())
    original=dict(critic.named_parameters())
    for name,p in compiled.named_parameters():p.requires_grad_(original[name].requires_grad)
    compiled.train(critic.training)
    return compiled
