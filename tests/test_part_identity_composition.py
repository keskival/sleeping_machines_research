import torch
from sleeping_machines.part_identity_composition import shape_signature,fit_shape_vocabulary,PartIdentityComposition
from sleeping_machines.observed_ownership_composition import ObservedOwnershipComposition


def fixture():
    return torch.tensor([[0.,0,0,1],[.01,0,5,-1],[.02,5,0,1],[.03,0,28,1],[.04,0,33,-1],[.05,4,29,1]],dtype=torch.float64)


def test_observable_identity_is_invariant_to_translation_time_and_row_order():
    x=fixture()[:3];y=x.flip(0)+torch.tensor([71.,9.,-17.,0.])
    assert shape_signature(x)==shape_signature(y)
    y[0,3]*=-1;assert shape_signature(x)!=shape_signature(y)


def test_teacher_does_not_change_native_forward_and_trains_the_actual_part_memories():
    x=fixture();vocab=fit_shape_vocabulary([x],10.7);assert len(vocab)==2
    args=dict(payload=4,heads=2,levels=7,depth=4,classes=2,ownership_threshold=10.7)
    torch.manual_seed(1423);base=ObservedOwnershipComposition(**args).double()
    torch.manual_seed(1423);model=PartIdentityComposition(**args,shape_vocabulary=vocab).double();model.load_acquired(base.state_dict())
    def run(actor):
        torch.manual_seed(1427);state=actor.new_state()
        for time,y,x,p in fixture():actor.observe(state,(int(y),int(x)),float(time),[float(p),1.])
        z,_=actor.compose(state);return z,state
    a,_=run(base);b,state=run(model);assert torch.equal(a,b)
    counts=(state.deliveries,state.publications,state.timer_emissions)
    loss,record=model.identity_loss(state,x);loss.backward()
    assert record['parts']==2 and model.identity_head.weight.grad.norm()>0
    assert model.objects[1].core.content.weight.grad.norm()>0
    assert counts==(state.deliveries,state.publications,state.timer_emissions)
    assert model.work(state)['identity_head_madds']==2*model.width*len(vocab)
