"""B1/R1 exact decision-preserving lazy evaluation of a bounded correction."""
from dataclasses import dataclass
import torch
@dataclass
class BoundPacket:
    value:float
    radius:float
    head_version:int

def bound_packet(head,radius,head_version):
    first=head.net[0].weight.detach()[:,-4:];last=head.net[-1].weight.detach().flatten()
    lipschitz=float(head.scale*last.norm()*torch.linalg.matrix_norm(first,ord=2)*radius)
    bounded=float(2*head.scale*last.abs().sum());return BoundPacket(min(lipschitz,bounded)*(1+1e-12)+1e-15,radius,head_version)

def select(base,z,actions,head,alpha,packet,head_version):
    if packet.head_version!=head_version:raise ValueError('Stale correction bound')
    if float(actions.norm(dim=-1).max())>packet.radius+1e-12:raise ValueError('Action exceeds certified radius')
    if alpha<0 or alpha>1:raise ValueError('Unsupported trust coefficient')
    ordered=base.flatten(1).sort(1).values;gap=ordered[:,1]-ordered[:,0];need=(gap<=2*alpha*packet.value) if alpha else torch.zeros(len(base),dtype=torch.bool)
    scores=base.clone()
    if bool(need.any()):scores[need]+=alpha*head.values(z[need],actions[need])
    return scores.flatten(1).argmin(1),need
