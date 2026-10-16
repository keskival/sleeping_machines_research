import pytest
import torch
from tests.test_bound_native_composition import actors,run
from sleeping_machines.bound_recursive_race_learning import BoundRecursiveRaceHeads


def test_recursive_race_can_pool_the_actual_bound_memories_with_live_credit():
    _,actor=actors();_,_,state=run(actor);parts=actor.parts(state,3,(0,0))
    head=BoundRecursiveRaceHeads(actor.width,actor.levels,slots=len(parts),identities=3,hidden=8).double()
    counts=(state.deliveries,state.publications,state.timer_emissions)
    ledger,work=head.forecast_parts(actor,state,3,(0,0))
    assert work['scored_race_slots']==len(parts)
    loss=ledger.silence(ledger.cursor+1)+ledger.event(ledger.cursor+2,torch.tensor([1]))[0]
    loss.sum().backward()
    assert head.heads[3].out.weight.grad.norm()>0
    assert actor.objects[2].core.content.weight.grad.norm()>0
    assert counts==(state.deliveries,state.publications,state.timer_emissions)


def test_pool_declaration_cannot_silently_drop_a_bound_child():
    _,actor=actors();_,_,state=run(actor);n=len(actor.parts(state,3,(0,0)))
    head=BoundRecursiveRaceHeads(actor.width,actor.levels,slots=n+1,identities=3,hidden=8).double()
    with pytest.raises(ValueError):head.forecast_parts(actor,state,3,(0,0))
