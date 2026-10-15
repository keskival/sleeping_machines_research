import torch
from sleeping_machines.carried_role_composition import CarriedRoleComposition
from sleeping_machines.relational_frame_composition import RelationalFrameComposition


def construct(gain):
    torch.manual_seed(71)
    return CarriedRoleComposition(payload=4,heads=2,levels=4,depth=4,classes=2,carrier_gain=gain).double()


def run(model):
    torch.manual_seed(73);state=model.new_state()
    for i,xy in enumerate([(0,0),(0,6),(6,0),(6,6)]):model.observe(state,xy,i*.01,[(-1.)**i,1.])
    return model.compose(state)[0],state


def test_zero_carrier_exact_migration_and_forward():
    torch.manual_seed(71);old=RelationalFrameComposition(payload=4,heads=2,levels=4,depth=4,classes=2).double()
    new=construct(0);new.load_relational_state(old.state_dict())
    x,_=run(old);y,state=run(new)
    assert torch.equal(x,y)
    assert new.work(state)['carrier_transport_calls']>0


def test_live_carrier_preserves_native_and_interaction_gradients():
    model=construct(.1);output,state=run(model);output.square().sum().backward()
    assert all(obj.core.content.weight.grad.norm()>0 for obj in model.objects)
    assert all(obj.carrier_scale.grad is not None and torch.isfinite(obj.carrier_scale.grad) for obj in model.objects[1:])
    root=state.banks[-1];private=root.private[(0,0)];role,time=private.role_carrier
    assert role.requires_grad and float(time.detach())<=state.now
    assert model.objects[-1].core.composer.pair_weight.grad.norm()>0


def test_indexed_read_carrier_jacobian_has_direct_transport_path():
    model=construct(.1);_,state=run(model);obj=model.objects[-1];bank=state.banks[-1]
    private=bank.private[(0,0)];value,time=private.role_carrier
    # Isolate the new carrier path from native memory's existing dependence.
    probe=value.detach().clone().requires_grad_();private.role_carriers[bank.published[(0,0)].version]=(probe,time)
    query_time=state.now
    result,_=obj.read(bank,(0,0),query_time)
    grad=torch.autograd.grad(result.sum(),probe)[0]
    assert torch.isfinite(grad).all() and grad.norm()>0
    with torch.no_grad():obj.carrier_scale.zero_()
    result,_=obj.read(bank,(0,0),query_time)
    assert torch.equal(torch.autograd.grad(result.sum(),probe)[0],torch.zeros_like(probe))


def test_carrier_reads_only_published_version_not_private_future():
    model=construct(.1);_,state=run(model);obj=model.objects[-1];bank=state.banks[-1]
    private=bank.private[(0,0)];version=bank.published[(0,0)].version
    first,_=obj.read(bank,(0,0),state.now)
    role,time=private.role_carriers[version]
    private.role_carriers[version+1]=(role*100000,time+1.)
    second,_=obj.read(bank,(0,0),state.now)
    assert torch.equal(first,second)
