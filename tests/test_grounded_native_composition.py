import torch
from sleeping_machines.grounded_native_composition import ObservedGeometrySketch,GroundedNativeComposition
from sleeping_machines.relational_frame_composition import RelationalFrameComposition


def test_observed_sketch_translation_and_time_nuisance_invariance():
    sketch=ObservedGeometrySketch().double()
    events=torch.tensor([[1.,3.,4.,1.],[2.,5.,6.,-1.]],dtype=torch.float64)
    x,n=sketch(events,3,(0,0),(2,2));shift=events+torch.tensor([123.,17.,19.,0.])
    y,m=sketch(shift,3,(0,0),(19,21))
    assert n==m==2 and torch.equal(x,y)


def test_geometry_retains_position_polarity_and_support_boundaries():
    sketch=ObservedGeometrySketch().double()
    events=torch.tensor([[0.,0.,0.,1.],[1.,3.,3.,-1.]],dtype=torch.float64)
    a,n=sketch(events,2,(0,0),(0,0))
    swapped=events.clone();swapped[:,3]*=-1;b,_=sketch(swapped,2,(0,0),(0,0))
    assert n==2 and not torch.allclose(a,b)
    _,n=sketch(events,1,(0,0),(0,0));assert n==1


def run(model):
    torch.manual_seed(923);events=[];state=model.new_state()
    for i,xy in enumerate([(0,0),(0,6),(6,0),(6,6)]):
        model.observe(state,xy,i*.01,[(-1.)**i,1.]);events.append([i*.01,*xy,(-1.)**i])
    logits,_=model.compose(state);return logits,state,events


def test_teacher_never_changes_actor_forward_and_reaches_every_level():
    args=dict(payload=4,heads=2,levels=4,depth=4,classes=2)
    torch.manual_seed(919);base=RelationalFrameComposition(**args).double()
    torch.manual_seed(919);model=GroundedNativeComposition(**args).double();model.load_relational_state(base.state_dict())
    original,_,_=run(base);output,state,events=run(model)
    assert torch.equal(original,output)
    loss,record=model.grounding_loss(state,events);loss.backward()
    assert len(record['level_losses'])==4 and all(n>0 for n in record['objects_per_level'])
    assert all(obj.core.content.weight.grad.norm()>0 for obj in model.objects)
    assert not model.geometry.directions.requires_grad
    assert model.work(state)['grounding_trig_evaluations']>0
