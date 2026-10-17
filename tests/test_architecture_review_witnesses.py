"""Small mathematical witnesses for concrete R1 review findings, not fits."""
import copy
import torch
from sleeping_machines.local_deep_race import LocalDeepRace
from sleeping_machines.packed_local_deep_race import PackedLocalDeepRace


def test_nonlinear_alternative_can_reverse_linearized_route_credit():
    # Two legal alternatives; a nonlinear suffix where local tangent ranks
    # the second alternative in the opposite direction to actual task loss.
    theta=torch.tensor(0.,dtype=torch.float64,requires_grad=True)
    winner=torch.tensor(.2,dtype=torch.float64,requires_grad=True)
    alternative=torch.tensor(2.,dtype=torch.float64)
    loss=lambda x:(x*x-1).square()
    tangent=torch.autograd.grad(loss(winner),winner)[0]*(alternative-winner)
    actual=loss(alternative)-loss(winner)
    probability=theta.sigmoid()
    exact=torch.autograd.grad((1-probability)*loss(winner)+probability*loss(alternative),theta)[0]
    assert float(tangent)<0<float(actual) and float(exact)>0


def test_retained_rls_statistics_are_not_invariant_to_encoder_coordinate_drift():
    # An exact orthogonal encoder rotation leaves information unchanged.
    # The head and information matrix must rotate with it to preserve values.
    x=torch.tensor([[1.,0.],[0.,2.]],dtype=torch.float64)
    y=torch.tensor([1.,-1.],dtype=torch.float64);ridge=.1
    rotation=torch.tensor([[0.,-1.],[1.,0.]],dtype=torch.float64)
    covariance=torch.linalg.inv(ridge*torch.eye(2,dtype=torch.float64)+x.T@x)
    response=covariance@x.T@y
    moved=x@rotation.T
    moved_covariance=torch.linalg.inv(ridge*torch.eye(2,dtype=torch.float64)+moved.T@moved)
    moved_response=moved_covariance@moved.T@y
    assert torch.allclose(moved_response,rotation@response)
    assert torch.allclose(moved_covariance,rotation@covariance@rotation.T)
    assert not torch.allclose(moved@response,x@response)
    assert torch.allclose(moved@moved_response,x@response)


def test_current_message_has_forward_sensitivity_without_lower_mix_credit():
    # This is the direct inter-layer path of the packed native kernel. Query
    # credit is a separate path and cannot make this detached edge exact.
    torch.manual_seed(17171)
    core=PackedLocalDeepRace(LocalDeepRace(8,depth=3,heads=2,pool=2))
    message=torch.randn(8);probe=torch.randn(8)
    def forward():
        values,state,packets,report=core.consume(message,core.new_state(),torch.Generator().manual_seed(71),torch.Generator().manual_seed(72),learn=True)
        return values[-1]@probe,report
    objective,report=forward();gradient=torch.autograd.grad(objective,core.mix)[0]
    assert torch.equal(gradient[0],torch.zeros_like(gradient[0]))
    original=core.mix.detach().clone();direction=torch.randn_like(original[0]);epsilon=1e-3
    with torch.no_grad():core.mix[0].copy_(original[0]+epsilon*direction)
    plus,pr=forward()
    with torch.no_grad():core.mix[0].copy_(original[0]-epsilon*direction)
    minus,mr=forward()
    with torch.no_grad():core.mix.copy_(original)
    assert pr['winners']==report['winners']==mr['winners']
    assert abs(float((plus-minus).detach())/(2*epsilon))>1e-4
