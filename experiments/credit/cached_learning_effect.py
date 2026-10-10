"""B1/R1 algebraically unchanged correction with reused context projection."""
import torch
from torch.nn import functional as F

def values(head,z,actions):
    first,last=head.net[0],head.net[-1];width=z.shape[-1]
    common=F.linear(z,first.weight[:,:width],first.bias)
    delta=F.linear(actions,first.weight[:,width:],None)
    difference=(common[:,:,None,:]+delta).tanh()-common.tanh()[:,:,None,:]
    result=head.scale*F.linear(difference,last.weight,None).squeeze(-1)
    # Keep zero bias gradient so optimizer decay/momentum semantics match the
    # original subtraction of two biased outputs, including nonzero biases.
    return result+0*last.bias.sum()
