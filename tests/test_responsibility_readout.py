import torch
from sleeping_machines.responsibility_readout import ResponsibilityReadout
from sleeping_machines.responsibility_native_composition import ResponsibilityNativeComposition


def test_no_event_count_bias_and_no_forced_depth_migration():
    torch.manual_seed(3);model=ResponsibilityReadout(3,4,2).double()
    views=torch.ones(3,4,dtype=torch.float64)
    logits,record=model(views)
    logits.square().sum().backward()
    assert torch.allclose(model.gate.grad,torch.zeros_like(model.gate.grad),atol=1e-12)
    assert record['weights'][0]>record['weights'][1]>record['weights'][2]


def test_prediction_gradient_moves_allocation_to_useful_higher_view():
    model=ResponsibilityReadout(2,2,2,initial_slope=0).double()
    with torch.no_grad():model.head.weight.copy_(torch.eye(2));model.head.bias.zero_()
    logits,record=model(torch.tensor([[2.,0.],[0.,2.]],dtype=torch.float64))
    torch.nn.functional.cross_entropy(logits[None],torch.tensor([1])).backward()
    assert model.gate.grad[1]<0<model.gate.grad[0]


def test_direct_credit_survives_tiny_output_allocation_and_masks_missing():
    model=ResponsibilityReadout(3,4,2).double()
    with torch.no_grad():model.gate[2]=-100
    views=torch.randn(3,4,dtype=torch.float64,requires_grad=True)
    z,record=model(views);loss,_=model.objective(z,record,1);loss.backward()
    assert views.grad[2].norm()>1e-6
    _,record=model(views,torch.tensor([True,False,True]));assert record['weights'][1]==0


def test_frozen_gate_source_cut_matches_explicit_feature_removal():
    model=ResponsibilityReadout(3,4,2).double();_,record=model(torch.randn(3,4,dtype=torch.float64))
    cuts=model.gate_value_effect(record)
    for i in range(3):
        values=record['weights'][:,None]*record['values'];values=values.clone();values[i]=0
        assert torch.allclose(cuts[i],model.head(values.sum(0)))


def test_native_integrated_readout_has_all_level_credit_and_paid_reads():
    torch.manual_seed(5)
    model=ResponsibilityNativeComposition(payload=4,heads=2,levels=4,depth=4,classes=2).double()
    state=model.new_state()
    for i,xy in enumerate([(0,0),(0,6),(6,0),(6,6)]):model.observe(state,xy,i*.01,[(-1.)**i,1.])
    z,_=model.compose(state);loss,_=model.responsibility.objective(z,model.last_responsibility,1);loss.backward()
    assert all(obj.core.content.weight.grad.norm()>0 for obj in model.objects)
    assert model.work(state)['responsibility_reads']>0
    assert not state.queue and model.last_responsibility['weights'].sum().allclose(torch.tensor(1.,dtype=torch.float64))
