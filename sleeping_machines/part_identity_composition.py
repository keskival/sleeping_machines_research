"""TRAIN-observable intrinsic part identity teaches the native representation.

Signatures are teacher targets only. The actor never receives a signature/id
as input;inference is the existing native ownership/role hierarchy.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .observed_ownership_composition import ObservedOwnershipComposition,observed_groups


def shape_signature(events):
    rows=torch.as_tensor(events,dtype=torch.float64)
    if rows.ndim!=2 or rows.shape[1]!=4 or not len(rows) or not bool(torch.isfinite(rows).all()):
        raise ValueError('finite revealed time,y,x,polarity rows required')
    if not torch.equal(rows[:,1:3],rows[:,1:3].round()):raise ValueError('integer pixel geometry required')
    xy=rows[:,1:3]-rows[:,1:3].min(0).values
    return tuple(sorted((int(p[0]),int(p[1]),1 if row[3]>0 else -1) for p,row in zip(xy,rows)))


def fit_shape_vocabulary(examples,threshold):
    signatures=set()
    for events in examples:
        rows=torch.as_tensor(events,dtype=torch.float64)
        groups,_=observed_groups([tuple(map(int,p)) for p in rows[:,1:3]],threshold)
        for group in groups:
            support=set(group);keep=[tuple(map(int,p)) in support for p in rows[:,1:3]]
            signatures.add(shape_signature(rows[keep]))
    if len(signatures)<2:raise ValueError('at least two observed shapes required')
    return {sig:i for i,sig in enumerate(sorted(signatures))}


class PartIdentityComposition(ObservedOwnershipComposition):
    def __init__(self,*args,shape_vocabulary,**kwargs):
        super().__init__(*args,**kwargs)
        if set(shape_vocabulary.values())!=set(range(len(shape_vocabulary))):raise ValueError('contiguous TRAIN vocabulary required')
        self.shape_vocabulary=dict(shape_vocabulary)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1409);self.identity_head=nn.Linear(self.width,len(shape_vocabulary))

    def load_acquired(self,weights):
        missing=self.load_state_dict(weights,strict=False)
        if missing.unexpected_keys or set(missing.missing_keys)!={'identity_head.weight','identity_head.bias'}:
            raise ValueError('native migration coverage changed')

    def identity_loss(self,state,events):
        rows=torch.as_tensor(events,dtype=torch.float64)
        if state.queue or state.ownership is None or len(rows)!=sum(state.input_issued.values()) or float(rows[:,0].max())>state.now:
            raise ValueError('identity supervision requires exactly the revealed drained scene')
        losses=[];correct=0;count=0;signatures=[]
        for address in sorted({a for a,_ in state.ownership.values()}):
            pixels={p for p,(a,_) in state.ownership.items() if a==address}
            keep=[tuple(map(int,p)) in pixels for p in rows[:,1:3]]
            sig=shape_signature(rows[keep])
            if sig not in self.shape_vocabulary:raise ValueError('unseen shape is not an admitted teacher target')
            label=self.shape_vocabulary[sig];value,_=self.objects[1].read(state.banks[1],address,state.now)
            logits=self.identity_head(value)
            losses.append(F.cross_entropy(logits[None],torch.tensor([label],device=logits.device)))
            correct+=int(logits.detach().argmax()==label);count+=1;signatures.append(label)
        if not losses:raise ValueError('no complete observed parts')
        state.identity_work=dict(identity_reads=count,identity_head_madds=count*self.width*len(self.shape_vocabulary),identity_target_event_tests=count*len(rows))
        return torch.stack(losses).mean(),dict(correct=correct,parts=count,shape_labels=signatures)

    def work(self,state):
        result=super().work(state);result.update(getattr(state,'identity_work',{}));return result
