import io
import pytest
import torch
from sleeping_machines.identity_role_composition import IdentityRoleCore,IdentityRoleComposition
from sleeping_machines.relational_frame_composition import RelationalEventCore
from sleeping_machines.indexed_temporal_object import TensorTimedEventCore
from sleeping_machines.part_identity_composition import fit_shape_vocabulary


def events():
    return torch.tensor([[0.,0,0,1],[.01,0,5,-1],[.02,5,0,1],[.03,0,28,1],[.04,0,33,-1],[.05,4,29,1]],dtype=torch.float64)


def actor():
    torch.manual_seed(1621)
    return IdentityRoleComposition(payload=4,heads=2,levels=7,depth=4,classes=2,ownership_threshold=10.7,shape_vocabulary=fit_shape_vocabulary([events()],10.7)).double()


def test_same_role_preserves_joint_content_pose_and_is_enumeration_invariant():
    model=actor();core=model.objects[2].core
    a=torch.arange(8,dtype=torch.float64,requires_grad=True);b=torch.arange(8,0,-1,dtype=torch.float64,requires_grad=True)
    entries=[((1,(0,0)),a,(.1,.1)),((1,(0,1)),b,(.4,.3))]
    x=core.relation(entries);y=core.relation(list(reversed(entries)))
    assert torch.allclose(x,y,atol=1e-12,rtol=1e-12)
    swapped=core.relation([(entries[0][0],b,entries[0][2]),(entries[1][0],a,entries[1][2])])
    assert not torch.allclose(x,swapped)
    x.square().sum().backward();assert a.grad.norm()>0 and b.grad.norm()>0


def test_native_actor_teaches_actual_issued_part_versions_and_deep_credit():
    model=actor();state=model.new_state();torch.manual_seed(1627)
    for t,y,x,p in events():model.observe(state,(int(y),int(x)),float(t),[float(p),1.])
    z,_=model.compose(state);loss,record=model.identity_loss(state,events())
    assert record['parts']>=2 and record['endpoint']=='parent-used immutable versions at issue time'
    (z.square().sum()+loss).backward()
    assert all(o.core.content.weight.grad is not None and o.core.content.weight.grad.norm()>0 for o in model.objects)
    work=model.work(state);assert work['identity_relation_pairs']>0 and work['identity_role_slots']>0
    with pytest.raises(ValueError):model.read_bound(state,next(p.source for e in state.bindings.entries.values() for p in e.parts if p.source[0]==1),-1.)


def test_inflight_inference_checkpoint_rekeys_envelopes_exactly():
    model=actor().eval();torch.manual_seed(1631)
    with torch.no_grad():
        state=model.new_state()
        for t,y,x,p in events():model.observe(state,(int(y),int(x)),float(t),[float(p),1.])
        model.compose(state)
        address=next(iter(state.banks[1].published));target=tuple(v//2 for v in address)
        model.query(state,1,address,2,target)
        assert state.queue and state.bindings.inflight
        buf=io.BytesIO();torch.save(model.checkpoint_scene(state),buf);rng=torch.get_rng_state()
        model.drain(state);expected=model.objects[2].read(state.banks[2],target,state.now)[0]
        buf.seek(0);restored=model.restore_scene(torch.load(buf,weights_only=False));torch.set_rng_state(rng)
        model.drain(restored);actual=model.objects[2].read(restored.banks[2],target,restored.now)[0]
        assert torch.equal(expected,actual) and state.now==restored.now


def test_same_role_children_do_not_overwrite_and_refresh_keeps_identity():
    from sleeping_machines.bound_native_composition import PartBinding
    model=actor();core=model.objects[2].core;state=core.new_state()
    for i,address in enumerate([(0,0),(0,1),(0,0)]):
        time=torch.tensor(float(i),dtype=torch.float64)
        state.current_part=PartBinding((1,address,i+1),time,(.1,.1),'query')
        mark=torch.cat((torch.arange(8,dtype=torch.float64)+i,torch.tensor([.1,.1,1/7,0.,1.,0.],dtype=torch.float64)))
        core.consume_event(0,time,mark,state)
    assert len(state.participants)==2
    assert state.participants[(1,(0,0))][-1]==(1,(0,0),3)
    assert state.identity_relation_reads==5
