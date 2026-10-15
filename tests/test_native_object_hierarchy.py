import torch
from sleeping_machines.native_object_hierarchy import NativeObjectHierarchy


def make():
    torch.manual_seed(7)
    return NativeObjectHierarchy(input_width=2,payload=4,heads=2,levels=8,depth=4,pool=2,classes=3).double()


def test_native_events_build_eight_levels_without_more_observation_history():
    model=make();state=model.new_state()
    model.observe(state,(0,0),0.,torch.tensor([1.,-.5],dtype=torch.double))
    model.advance(state,1.)
    assert len(state.inputs)==1 and state.publications==8
    assert all((0,0) in bank.published for bank in state.banks)
    assert sum(bank.private[(0,0)].selected_updates for bank in state.banks)==8*4*2
    loss=model.logits(state).square().sum();loss.backward()
    assert torch.isfinite(model.input.weight.grad).all() and model.input.weight.grad.abs().sum()>0
    assert any(p.grad is not None and p.grad.abs().sum()>0 for n,p in model.objects[0].named_parameters() if 'clock_bias' in n)


def test_queries_and_generated_events_preserve_indexed_input_and_memory():
    model=make();state=model.new_state();model.observe(state,(0,0),0.,[1.,.2]);model.advance(state,1.)
    value,version=model.read(state,0,(0,0));clone=value.clone()
    published_before=state.banks[0].published[(0,0)]
    model.query(state,0,(0,0),7,(0,0));model.advance(state,1.2)
    assert len(state.inputs)==1 and state.queries==1
    assert state.banks[0].published[(0,0)] is published_before
    assert model.read(state,0,(0,0))[1]==version
    assert not torch.equal(model.read(state,0,(0,0))[0],clone) # lazy evolution, no write
    model.arm(state,0,(0,0));model.advance(state,1.5)
    assert state.timer_emissions==1
    assert state.banks[0].published[(0,0)] is published_before
    assert not state.queue # no uncontrolled self-excitation/rearming


def test_future_observation_never_enters_current_index():
    model=make();state=model.new_state()
    model.observe(state,(0,0),2.,[1.,0.]);model.advance(state,1.)
    assert not state.inputs and state.publications==0
    model.advance(state,3.)
    assert len(state.inputs)==1 and state.publications==8


def test_input_versions_follow_event_time_rather_than_enqueue_order():
    model=make();state=model.new_state()
    model.observe(state,(0,0),2.,[2.,0.]);model.observe(state,(0,0),1.,[1.,0.])
    model.advance(state,3.)
    assert state.inputs[(0,0)][1:]==(2,2.)
    work=model.work(state)
    assert work['external_observations']==2
    assert work['selected_updates']==2*8*4*2
    assert work['candidate_scores']==work['selected_updates']*2
