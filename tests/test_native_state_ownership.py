"""A parent hypothesis must not overwrite the shared parts it interprets."""
import torch
from sleeping_machines.native_memory_composition import NativeMemoryComposition


def test_parent_state_cannot_feed_back_into_next_lower_representation_update():
    torch.manual_seed(151)
    model=NativeMemoryComposition(input_width=2,payload=4,heads=2,levels=8,depth=4).double()
    def first():
        state=model.new_state();model.observe(state,(0,0),0.,[1.,1.]);model.drain(state);return state
    torch.manual_seed(157);a=first()
    torch.manual_seed(157);b=first()
    parent=b.banks[-1].private[(0,0)]
    parent.memories={key:value*3+.7 for key,value in parent.memories.items()}
    values,times=parent.contexts[0];parent.contexts[0]=(values*3+.7,times)
    def next_event(state):
        torch.manual_seed(163)
        model.observe(state,(0,0),10.,[-1.,1.]);model.drain(state)
    next_event(a);next_event(b)
    for aa,bb in zip(a.banks[:-1],b.banks[:-1]):
        assert torch.equal(aa.published[(0,0)].values,bb.published[(0,0)].values)
    assert not torch.equal(a.banks[-1].published[(0,0)].values,b.banks[-1].published[(0,0)].values)
