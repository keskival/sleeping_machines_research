import pytest
import torch
from sleeping_machines.split_event_heads import SplitEventHeads
from sleeping_machines.indexed_temporal_object import TensorTimedEventCore,IndexedTemporalObject


def object_and_state():
    torch.manual_seed(11)
    model=IndexedTemporalObject(content_dim=8,payload=4,heads=2,depth=2,protected_pairs=1).double()
    return model,model.new_state()


def test_tensor_time_adapter_preserves_native_scalar_transition_bitwise():
    torch.manual_seed(1)
    kw=dict(sources=1,content_dim=8,classes=8,payload=4,heads=2,depth=2,shared_maps=True,protected_pairs=1)
    original=SplitEventHeads(**kw).double();adapted=TensorTimedEventCore(**kw).double()
    adapted.load_state_dict(original.state_dict());a=original.new_state();b=adapted.new_state()
    for step,time in enumerate([0.,.1,.4]):
        content=torch.arange(8,dtype=torch.double)/8
        torch.manual_seed(31+step);ya,ta=original.consume_event(0,time,content,a)
        torch.manual_seed(31+step);yb,tb=adapted.consume_event(0,torch.tensor(time,dtype=torch.double),content,b)
        assert torch.equal(ya,yb) and torch.equal(ta,tb)
        assert a.memories.keys()==b.memories.keys()
        for k in a.memories:assert torch.equal(a.memories[k],b.memories[k])


def test_unpublished_work_cannot_replace_visible_memory():
    model,state=object_and_state();content=torch.randn(8,dtype=torch.double)
    first=model.accept(state,'part',0.,content,('child',0))
    with pytest.raises(ValueError):model.publish(state,first,0.)
    model.publish(state,first,first.ready)
    before,_=model.read(state,'part',.05)
    pending=model.accept(state,'part',.1,content*2,('child',1))
    unchanged,version=model.read(state,'part',.05)
    assert version==1 and torch.equal(before,unchanged)
    model.publish(state,pending,pending.ready)
    assert state.published['part'].version==2


def test_read_is_persistent_lazy_and_does_not_add_evidence():
    model,state=object_and_state()
    p=model.accept(state,4,0.,torch.randn(8,dtype=torch.double),('source',0));model.publish(state,p,p.ready)
    snapshot={k:v.clone() for k,v in state.private[4].memories.items()}
    events=state.private[4].events
    a,_=model.read(state,4,.1);b,_=model.read(state,4,.2);repeat,_=model.read(state,4,.1)
    assert torch.equal(a,repeat) and not torch.equal(a,b)
    assert state.private[4].events==events
    assert all(torch.equal(v,snapshot[k]) for k,v in state.private[4].memories.items())


def test_retry_bookkeeping_is_bounded_by_streams_not_event_count():
    model,state=object_and_state()
    for version in range(5):
        p=model.accept(state,0,float(version),torch.ones(8,dtype=torch.double),('source',version))
        model.publish(state,p,p.ready)
        assert model.accept(state,0,float(version),torch.ones(8,dtype=torch.double),('source',version)) is None
    assert len(state.seen)==1 and state.private[0].events==5


def test_clock_speed_changes_generated_content_with_same_winner():
    model,state=object_and_state()
    p=model.accept(state,0,0.,torch.randn(8,dtype=torch.double),('source',0));model.publish(state,p,p.ready)
    torch.manual_seed(51);slow=model.arm(state,0,.2)
    slow_value,_=model.read(state,0,slow.due)
    with torch.no_grad():
        for unit in model.core.units[0][0][0]:unit.clock_bias.add_(3.)
    torch.manual_seed(51);fast=model.arm(state,0,.2)
    fast_value,_=model.read(state,0,fast.due)
    assert slow.winner==fast.winner and fast.due<slow.due
    assert not torch.allclose(slow_value,fast_value,atol=1e-10,rtol=1e-10)
    assert model.fire(state,slow,slow.due) is None # superseded
    assert model.fire(state,fast,fast.due) is not None
    assert model.fire(state,fast,fast.due) is None # consumed once


def test_incoming_time_remains_differentiable_through_native_memory():
    model,state=object_and_state()
    p=model.accept(state,0,0.,torch.randn(8,dtype=torch.double),('source',0));model.publish(state,p,p.ready)
    time=torch.tensor(.1,dtype=torch.double,requires_grad=True)
    p=model.accept(state,0,time,torch.randn(8,dtype=torch.double),('source',1));model.publish(state,p,p.ready)
    value,_=model.read(state,0,p.ready+.01);value.square().sum().backward()
    assert torch.isfinite(time.grad) and time.grad.abs()>0


def test_unpublished_future_write_cannot_change_current_timer():
    model,state=object_and_state();x=torch.randn(8,dtype=torch.double)
    p=model.accept(state,0,0.,x,('source',0));model.publish(state,p,p.ready)
    torch.manual_seed(61);before=model.arm(state,0,.05)
    model.accept(state,0,.2,x*10,('source',1))
    torch.manual_seed(61);after=model.arm(state,0,.05)
    assert torch.equal(before.due,after.due)


def test_generated_emission_loss_trains_the_clock_through_elapsed_time():
    model,state=object_and_state()
    p=model.accept(state,0,0.,torch.randn(8,dtype=torch.double),('source',0));model.publish(state,p,p.ready)
    timer=model.arm(state,0,.1);value,_=model.fire(state,timer,timer.due)
    value.square().sum().backward()
    gradients=[u.clock_bias.grad for u in model.core.units[0][0][0]]
    assert any(g is not None and torch.isfinite(g) and g.abs()>0 for g in gradients)
