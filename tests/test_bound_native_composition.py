import pytest
import torch
from sleeping_machines.bound_native_composition import BoundNativeComposition,BindingStore,BoundSnapshot,PartBinding
from sleeping_machines.relational_frame_composition import RelationalFrameComposition


def actors():
    args=dict(payload=4,heads=2,levels=4,depth=4,classes=2)
    torch.manual_seed(1103);base=RelationalFrameComposition(**args).double()
    torch.manual_seed(1103);bound=BoundNativeComposition(**args).double()
    assert set(base.state_dict())==set(bound.state_dict())
    return base,bound


def run(model):
    torch.manual_seed(1117);state=model.new_state()
    for i,xy in enumerate([(0,0),(0,6),(6,0),(6,6)]):model.observe(state,xy,i*.01,[(-1.)**i,1.])
    z,ready=model.compose(state)
    return z,ready,state


def test_native_forward_schedule_and_parameter_gradients_are_unchanged():
    base,bound=actors();a,ta,sa=run(base);b,tb,sb=run(bound)
    assert torch.equal(a,b) and torch.equal(ta,tb)
    assert (sa.deliveries,sa.publications,sa.timer_emissions)==(sb.deliveries,sb.publications,sb.timer_emissions)
    a.square().sum().backward();b.square().sum().backward()
    for (na,pa),(nb,pb) in zip(base.named_parameters(),bound.named_parameters()):
        assert na==nb
        assert (pa.grad is None and pb.grad is None) or torch.equal(pa.grad,pb.grad),na


def test_every_bound_child_resolves_to_an_available_lower_version():
    _,model=actors();_,_,state=run(model)
    assert not state.bindings.inflight
    root=model.parts(state,3,(0,0));assert root
    for e in state.bindings.entries.values():
        for p in e.parts:
            assert p.source[0]<e.key[0]
            child=state.bindings.entries[p.source]
            assert child.available<=p.issued<=e.available
    for level,bank in enumerate(state.banks):
        for address,record in bank.published.items():
            key=(level,address,record.version)
            expected,_=model.objects[level].read(bank,address,state.now)
            assert torch.equal(expected,model.read_bound(state,key,state.now))


def test_bound_version_survives_lower_write_and_collection_without_changing_parent():
    _,model=actors();_,_,state=run(model)
    parent=state.banks[1].published[(0,0)]
    parts=model.parts(state,1,(0,0));old=next(p.source for p in parts if p.source[0]==0)
    before=model.read_bound(state,old,state.now).clone()
    then=state.now
    model.observe(state,old[1],then+.1,[9.,1.]);model.drain(state)
    assert state.banks[0].published[old[1]].version>old[2]
    assert state.banks[1].published[(0,0)] is parent
    assert torch.equal(model.read_bound(state,old,then),before)
    model.collect_bindings(state)
    assert old in state.bindings.entries
    assert model.parts(state,1,(0,0))==parts


def test_bound_reads_are_not_events_and_credit_reaches_the_issued_child():
    _,model=actors();_,_,state=run(model)
    part=model.parts(state,3,(0,0))[0]
    counts=(state.deliveries,state.publications,state.timer_emissions,state.queries)
    a=model.read_bound(state,part.source,state.now)
    assert torch.equal(a,model.read_bound(state,part.source,state.now))
    a.square().sum().backward()
    assert model.objects[part.source[0]].core.content.weight.grad.norm()>0
    assert counts==(state.deliveries,state.publications,state.timer_emissions,state.queries)
    with pytest.raises(ValueError):model.read_bound(state,part.source,-1.)


def test_invalid_forward_or_future_bindings_are_rejected():
    store=BindingStore();t=torch.tensor(1.,dtype=torch.float64);key=(0,(0,0),1)
    store.put(BoundSnapshot(key,t,None,torch.ones(2),()))
    with pytest.raises(ValueError):store.put(BoundSnapshot(key,t,None,torch.ones(2),()))
    p=PartBinding(key,t-.1,(0.,0.),'query')
    with pytest.raises(ValueError):store.put(BoundSnapshot((1,(0,0),1),t,None,None,(p,)))


def test_higher_native_memory_can_query_its_bound_old_child_after_a_lower_write():
    _,model=actors();_,_,state=run(model)
    old=next(p.source for p in model.parts(state,1,(0,0)) if p.source[0]==0)
    model.observe(state,old[1],state.now+.1,[7.,1.]);model.drain(state)
    count=sum(state.input_issued.values());child_version=state.banks[0].published[old[1]].version
    model.query_bound(state,old,1,(0,0));model.drain(state)
    binding=next(p for p in model.parts(state,1,(0,0)) if p.source[:2]==old[:2])
    assert binding.source==old and binding.kind=='query'
    assert state.banks[0].published[old[1]].version==child_version>old[2]
    assert sum(state.input_issued.values())==count
    with pytest.raises(ValueError):model.query_bound(state,old,0,old[1])
