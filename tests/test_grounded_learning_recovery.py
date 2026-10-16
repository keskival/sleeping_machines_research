import io
import numpy as np
import torch
from sleeping_machines.grounded_native_composition import GroundedNativeComposition
from experiments.hierarchy.grounded_composition_learning import paired_backward,score


class Scene:
    def __init__(self,events,label):self.events,self.label=events,label
    def model_input(self):return self.events.copy(),self.label


def examples():
    a=np.array([[0,0,0,1],[10000,6,6,-1]],dtype=np.int64)
    b=a.copy();b[:,2]=6-b[:,2]
    return [Scene(a,0),Scene(b,1)]


def make(arm):
    torch.manual_seed(971);model=GroundedNativeComposition(payload=4,heads=2,levels=4,depth=4,classes=2)
    model.learning_arm=arm
    if arm=='decoder_only':
        for name,p in model.named_parameters():p.requires_grad_(name.startswith(('head.','geometry_decoders.')))
    return model


def step(m,o):
    m.train(m.learning_arm!='decoder_only');o.zero_grad(set_to_none=True)
    loss,_=paired_backward(m,[(r,r.label) for r in examples()]);torch.nn.utils.clip_grad_norm_(m.parameters(),1.);o.step();return loss


def test_grounded_pair_exact_adam_recovery():
    a=make('joint');oa=torch.optim.AdamW(a.parameters(),lr=.001);torch.manual_seed(977);step(a,oa)
    buf=io.BytesIO();torch.save(dict(model=a.state_dict(),optimizer=oa.state_dict(),rng=torch.get_rng_state()),buf)
    expected=step(a,oa);expected_rng=torch.get_rng_state();buf.seek(0);loaded=torch.load(buf,weights_only=False)
    b=make('joint');ob=torch.optim.AdamW(b.parameters(),lr=.001);b.load_state_dict(loaded['model']);ob.load_state_dict(loaded['optimizer']);torch.set_rng_state(loaded['rng'])
    assert step(b,ob)==expected and torch.equal(torch.get_rng_state(),expected_rng)
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())


def test_decoder_only_changes_no_native_representation():
    m=make('decoder_only');score(m,examples());before=m.scored_features.clone()
    weights={k:v.clone() for k,v in m.state_dict().items() if not k.startswith(('head.','geometry_decoders.'))}
    optimizer=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=.001);step(m,optimizer)
    score(m,examples());assert torch.equal(before,m.scored_features)
    assert all(torch.equal(v,m.state_dict()[k]) for k,v in weights.items())
