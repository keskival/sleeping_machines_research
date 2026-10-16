import torch
from sleeping_machines.centered_geometry_grounding import GeometryInnovationScale,CenteredGroundedNativeComposition
from sleeping_machines.relational_frame_composition import RelationalFrameComposition


def test_constant_reference_has_unit_error_after_innovation_scaling():
    targets=torch.tensor([[2.,5.],[4.,7.]],dtype=torch.float64)
    center=targets.mean(0);variance=(targets-center).square().mean()
    norm=GeometryInnovationScale(1,2).double();norm.set_statistics(center[None],variance[None])
    scaled=norm(targets,0)
    assert torch.equal(scaled.mean(0),torch.zeros(2,dtype=torch.float64))
    assert scaled.square().mean()==1
    assert not norm.center.requires_grad and not norm.scale.requires_grad


def test_target_level_mean_is_not_a_learning_job_and_zero_variance_is_finite():
    norm=GeometryInnovationScale(2,4).double();norm.set_statistics(torch.ones(2,4)*100,torch.tensor([0.,.5]))
    assert torch.equal(norm(torch.ones(4)*100,0),torch.zeros(4,dtype=torch.float64))
    assert torch.isfinite(norm(torch.ones(4),0)).all()


def test_normalization_never_changes_native_inference_or_zeroes_local_credit():
    args=dict(payload=4,heads=2,levels=4,depth=4,classes=2)
    torch.manual_seed(1013);base=RelationalFrameComposition(**args).double()
    model=CenteredGroundedNativeComposition(**args).double();model.load_relational_state(base.state_dict())
    outputs=[]
    for actor in [base,model]:
        torch.manual_seed(1019);state=actor.new_state();events=[]
        for i,xy in enumerate([(0,0),(0,6),(6,0),(6,6)]):
            actor.observe(state,xy,i*.01,[(-1.)**i,1.]);events.append([i*.01,*xy,(-1.)**i])
        outputs.append(actor.compose(state)[0])
    assert torch.equal(*outputs)
    model.normalization.set_statistics(torch.randn(4,32),torch.ones(4)*.1)
    loss,_=model.grounding_loss(state,events);loss.backward()
    assert all(obj.core.content.weight.grad.norm()>0 for obj in model.objects)
