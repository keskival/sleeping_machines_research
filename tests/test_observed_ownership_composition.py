import pytest
import torch
from sleeping_machines.observed_ownership_composition import ObservedOwnershipComposition,observed_groups


def test_groups_are_observed_translation_invariant_and_permutation_invariant():
    points=[(0,0),(0,5),(5,0),(0,28),(0,33),(5,28)]
    a,work=observed_groups(points,10.7);b,other=observed_groups([(y+7,x-19) for y,x in reversed(points)],10.7)
    assert a==[[ (y-7,x+19) for y,x in group] for group in b]
    assert len(a)==2 and work==other==15


def test_native_owned_parts_are_complete_and_queryable():
    torch.manual_seed(1289)
    model=ObservedOwnershipComposition(payload=4,heads=2,levels=7,depth=4,classes=2,ownership_threshold=10.7).double()
    state=model.new_state()
    for i,xy in enumerate([(0,0),(0,5),(5,0),(0,28),(0,33),(5,28)]):model.observe(state,xy,i*.01,[(-1.)**i,1.])
    z,_=model.compose(state);assert torch.isfinite(z).all()
    assert model.work(state)['observed_part_memories']==2
    for dest in {a for a,_ in state.ownership.values()}:
        parts=model.parts(state,1,dest)
        assert len(parts)==3
        assert {p.source[1] for p in parts}=={p for p,(a,_) in state.ownership.items() if a==dest}
    z.square().sum().backward()
    assert all(o.core.content.weight.grad is not None and o.core.content.weight.grad.norm()>0 for o in model.objects)
    model.observe(state,(10,10),state.now+.1,[1.,1.]);model.drain(state)
    with pytest.raises(ValueError):model.establish_ownership(state)


def test_ownership_does_not_change_actor_parameter_keys():
    from sleeping_machines.bound_native_composition import BoundNativeComposition
    args=dict(payload=4,heads=2,levels=4,depth=4,classes=2)
    torch.manual_seed(1291);a=BoundNativeComposition(**args)
    torch.manual_seed(1291);b=ObservedOwnershipComposition(**args,ownership_threshold=10.7)
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
