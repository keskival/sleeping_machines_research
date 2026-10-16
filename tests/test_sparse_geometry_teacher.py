import numpy as np
import torch
from sleeping_machines.grounded_native_composition import ObservedGeometrySketch,GroundedNativeComposition
from sleeping_machines.sparse_geometry_teacher import sparse_geometry_targets,SparseGroundedNativeComposition


def test_all_sparse_targets_equal_direct_observation_teacher():
    teacher=ObservedGeometrySketch().double()
    rng=np.random.default_rng(983);xy=rng.integers(-3,125,(96,2));pol=rng.choice([-1,1],96)
    events=np.column_stack([np.arange(96)*.001,xy,pol]);origin=tuple(xy.min(0))
    targets,work=sparse_geometry_targets(events,8,origin,teacher.directions)
    for (level,address),(value,count) in targets.items():
        expected,n=teacher(events,level,address,origin)
        assert count==n and torch.allclose(value,expected,atol=1e-12,rtol=1e-12)
    assert work['sparse_teacher_observation_visits']==96


def test_repeated_events_need_no_more_rotations_or_transcendentals():
    teacher=ObservedGeometrySketch().double()
    events=np.array([[0,0,0,1],[.001,6,6,-1]],dtype=float)
    a,wa=sparse_geometry_targets(events,8,(0,0),teacher.directions)
    b,wb=sparse_geometry_targets(np.repeat(events,100,axis=0),8,(0,0),teacher.directions)
    for key,(value,n) in a.items():
        assert b[key][1]==100*n and torch.allclose(b[key][0],10*value,atol=1e-12)
    assert wa['sparse_teacher_trig_evaluations']==wb['sparse_teacher_trig_evaluations']
    assert wa['sparse_teacher_merge_edges']==wb['sparse_teacher_merge_edges']


def run(model):
    state=model.new_state();events=[];torch.manual_seed(991)
    for i,xy in enumerate([(0,0),(0,6),(6,0),(6,6)]):
        model.observe(state,xy,i*.01,[(-1.)**i,1.]);events.append([i*.01,*xy,(-1.)**i])
    z,_=model.compose(state);loss,record=model.grounding_loss(state,events)
    return z,loss,record,state


def test_integrated_teacher_preserves_predictions_losses_and_native_gradients():
    args=dict(payload=4,heads=2,levels=4,depth=4,classes=2)
    torch.manual_seed(987);a=GroundedNativeComposition(**args).double()
    b=SparseGroundedNativeComposition(**args).double();b.load_state_dict(a.state_dict())
    za,la,ra,sa=run(a);zb,lb,rb,sb=run(b)
    assert torch.equal(za,zb) and torch.allclose(la,lb,atol=1e-12,rtol=1e-12)
    assert ra['objects_per_level']==rb['objects_per_level']
    la.backward();lb.backward()
    for (_,x),(_,y) in zip(a.named_parameters(),b.named_parameters()):
        if x.grad is not None:assert torch.allclose(x.grad,y.grad,atol=1e-11,rtol=1e-11)
    assert b.work(sb)['sparse_teacher_observation_visits']==4
