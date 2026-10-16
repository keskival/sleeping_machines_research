import copy
import io
import pytest
import torch
from sleeping_machines.queryable_local_deep_race import QueryableLocalDeepRace


def setup(depth=3):
    torch.manual_seed(16301)
    model=QueryableLocalDeepRace(width=8,depth=depth,heads=2,pool=3,publication_capacity=2).double()
    return model,model.new_state(),torch.Generator().manual_seed(16401),torch.Generator().manual_seed(16402)


def test_bound_publication_read_matches_published_value_and_is_immutable():
    model,state,g,a=setup()
    _,state,_,_=model.consume(torch.randn(8,dtype=torch.float64),state,g,a)
    key=state['publications'][0][-1];entry=state['bindings'].entries[key]
    before=model.read_publication(state,key,entry.available,[],learn=False)
    assert torch.allclose(before,entry.native.key_view,atol=2e-12,rtol=2e-12)
    for _ in range(3):model.consume(torch.randn(8,dtype=torch.float64),state,g,a)
    # This pinned reference is immutable even if the live receiver advances/GC runs.
    state['bindings'].entries[key]=entry
    after=model.read_publication(state,key,entry.available,[],learn=False)
    assert torch.equal(before,after)
    with pytest.raises(ValueError):model.read_publication(state,key,entry.available-1,[],learn=False)
    assert entry.native.key_view.grad_fn is None and entry.native.residual.grad_fn is None


def test_old_query_read_credits_the_actual_earlier_receiver_write_without_recurrence():
    model,state,g,a=setup(depth=2)
    model.consume(torch.randn(8,dtype=torch.float64),state,g,a)
    key=state['publications'][0][-1];entry=state['bindings'].entries[key]
    # No graph from the original event is used by this later local loss.
    packets=[];model.zero_grad(set_to_none=True)
    value=model.read_publication(state,key,entry.available+2,packets,learn=True)
    (value*torch.arange(1,9,dtype=torch.float64)).sum().backward()
    model.deliver_credit(packets,state)
    assert all(model.units[0][h][u].input.weight.grad is not None and
        model.units[0][h][u].input.weight.grad.norm()>0 for h,u in enumerate(entry.native.winners))
    assert all(t.value.grad_fn is None for t in entry.native.receivers)


def test_joint_composer_has_cross_part_dependence_and_binding_roles_matter():
    model,_,_,_=setup()
    composer=model.assemblers[0]
    left=torch.randn(8,dtype=torch.float64,requires_grad=True)
    right=torch.randn(8,dtype=torch.float64,requires_grad=True)
    out=composer({0:left,1:right})
    assert not torch.allclose(out,composer({0:right,1:left}))
    grad=torch.autograd.grad(out[0],left,create_graph=True)[0]
    mixed=torch.autograd.grad(grad[0],right)[0]
    assert mixed.norm()>1e-8


def test_deep_query_credit_and_exact_numerical_resume():
    model,state,g,a=setup();packets=[]
    for _ in range(4):
        output,state,p,_=model.consume(torch.randn(8,dtype=torch.float64),state,g,a);packets.extend(p)
    output.square().mul(torch.arange(1,9,dtype=torch.float64)).sum().backward()
    model.deliver_credit(packets,state)
    assert state['memory_selected_reads']>0 and state['queried_historical_versions']>0
    assert state['memory_pair_products']>0 and state['memory_alternative_reads']>0
    assert all(m.pair_weight.grad is not None and m.pair_weight.grad.norm()>0 for m in model.assemblers)
    assert all(t.value.grad_fn is None for t in state['receivers'].values())
    saved=io.BytesIO();torch.save((state,g.get_state(),a.get_state()),saved)
    message=torch.randn(8,dtype=torch.float64)
    expected=model.consume(message,state,g,a,learn=False)[0]
    saved.seek(0);restored,gs,als=torch.load(saved,weights_only=False)
    g.set_state(gs);a.set_state(als)
    actual=model.consume(message,restored,g,a,learn=False)[0]
    assert torch.equal(expected,actual)
    assert model.storage(restored)['temporal_graph_entries']==0
