import torch
from sleeping_machines.coalesced_native_composition import CoalescedNativeComposition


def test_local_writes_persist_before_generated_child_composition():
    torch.manual_seed(41)
    model=CoalescedNativeComposition(input_width=2,payload=4,heads=2,levels=8,depth=4,classes=2).double()
    state=model.new_state()
    for i,(y,x) in enumerate([(0,0),(127,127),(0,127),(127,0)]):
        model.observe(state,(y,x),i*.01,[(-1.)**i,1.])
    model.drain(state)
    assert len(state.banks[0].published)==4
    assert all(not bank.published for bank in state.banks[1:])
    output,ready=model.compose(state)
    active_lower=sum(len(bank.published) for bank in state.banks[:-1])
    assert state.timer_emissions==active_lower+1
    assert state.queries==0 and sum(state.input_issued.values())==4
    assert state.deliveries==4+active_lower and not state.queue
    output.square().sum().backward()
    assert torch.isfinite(ready)
    assert all(obj.core.content.weight.grad.abs().sum()>0 for obj in model.objects)


def test_emission_policy_preserves_selected_only_stochastic_eval():
    model=CoalescedNativeComposition(input_width=2,payload=4,heads=2,levels=8,depth=4,classes=2).double()
    outputs=[]
    for training in [True,False]:
        model.train(training);torch.manual_seed(57);state=model.new_state()
        model.observe(state,(4,5),0.,[1.,1.]);model.observe(state,(123,119),.01,[-1.,1.])
        outputs.append(model.compose(state))
    assert all(torch.equal(a,b) for a,b in zip(*outputs))
