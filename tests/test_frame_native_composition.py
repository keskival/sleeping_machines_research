import torch
from sleeping_machines.frame_native_composition import FrameNativeComposition
from sleeping_machines.coalesced_native_composition import CoalescedNativeComposition


def execute(model,shift=(0,0)):
    state=model.new_state()
    for i,(y,x,pol) in enumerate([(3,4,1.),(30,90,-1.),(110,7,1.),(3,4,-1.)]):
        model.observe(state,(y+shift[0],x+shift[1]),i*.01,[pol,1.])
    output,time=model.compose(state)
    return state,output,time


def make():
    return FrameNativeComposition(input_width=2,payload=4,heads=2,levels=8,depth=4,classes=2).double()


def test_global_translation_is_exact_with_native_times_routes_and_gradients():
    torch.manual_seed(83);model=make()
    torch.manual_seed(89);a,ya,ta=execute(model)
    torch.manual_seed(89);b,yb,tb=execute(model,(7,-9))
    assert torch.equal(ya,yb) and torch.equal(ta,tb)
    assert tuple(v-w for v,w in zip(b.frame_origin,a.frame_origin))==(7,-9)
    assert len(a.banks[0].published)==3 # repeated pixel integrated persistently
    assert model.work(a)==model.work(b)
    for ba,bb in zip(a.banks[1:],b.banks[1:]):
        assert set(ba.published)==set(bb.published)
        assert all(torch.equal(ba.published[k].values,bb.published[k].values) for k in ba.published)
    ya.square().sum().backward()
    assert all(obj.core.content.weight.grad.abs().sum()>0 for obj in model.objects)


def test_frame_commit_does_not_replay_sensor_events_or_change_parameters():
    torch.manual_seed(97);base=CoalescedNativeComposition(input_width=2,payload=4,heads=2,levels=8,depth=4,classes=2).double()
    rng=torch.get_rng_state()
    torch.manual_seed(97);model=make();assert torch.equal(rng,torch.get_rng_state())
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in base.state_dict().items())
    state,_,_=execute(model)
    assert sum(state.input_issued.values())==4
    assert state.frame_origin==(3,4)
    original=state.frame_origin
    model.observe(state,(1,1),state.now+.01,[1.,1.]);model.drain(state)
    assert state.frame_origin==original # upper memories never silently reindexed
    assert (1,1) in state.banks[0].published
