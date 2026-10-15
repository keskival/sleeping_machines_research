"""Integrate the actual native backward over the exponential race law."""
from types import SimpleNamespace
import numpy as np
import torch
from sleeping_machines.sparse_race_language import TemporalRoute


def integrated_credit(power):
    scores=torch.tensor([.2,-.5,.7],dtype=torch.double,requires_grad=True)
    values=torch.tensor([[-1.],[.3],[2.]],dtype=torch.double)
    nodes,weights=np.polynomial.laguerre.laggauss(64)
    nodes=torch.from_numpy(nodes);weights=torch.from_numpy(weights)
    rates=scores.exp();pi=rates/rates.sum();times=nodes/rates.sum()
    delay=.001+.010*times/(1+times)
    coefficient=1+2*delay+delay.square()
    offset=delay.square()-.2*delay
    losses=values[:,0,None].pow(power)*coefficient[None]+offset[None]
    risk=(pi[:,None]*losses*weights[None]).sum()
    exact=torch.autograd.grad(risk,scores)[0]
    # Joint log p(J,T)=s_J-Lambda*T. Rao-Blackwellize J at each T.
    joint=(weights[None]*pi[:,None]*(losses-times[None]*rates.sum()*(pi[:,None]*losses).sum(0)[None])).sum(1)
    native=torch.zeros_like(scores)
    for n,time in enumerate(times.detach()):
        d=delay[n].detach();a=coefficient[n].detach()
        for winner in range(3):
            value=values[winner,0]
            error_value=(power*value.pow(power-1)*a).reshape(1)
            error_delay=value.pow(power)*(2+2*d)+(2*d-.2)
            ctx=SimpleNamespace(saved_tensors=(rates.detach(),time,torch.tensor(winner),values))
            score_gradient,_=TemporalRoute.backward(ctx,error_value,error_delay,None)
            native+=weights[n]*pi[winner].detach()*score_gradient
    return exact.detach(),joint.detach(),native


def test_native_clock_and_value_credit_are_jointly_exact_for_affine_value_loss():
    exact,joint,native=integrated_credit(1)
    assert torch.allclose(native,exact,atol=1e-11,rtol=1e-11)
    assert torch.allclose(joint,exact,atol=1e-11,rtol=1e-11)


def test_joint_hazard_audit_covers_nonlinear_loss_that_local_teacher_does_not():
    exact,joint,native=integrated_credit(2)
    assert torch.allclose(joint,exact,atol=1e-11,rtol=1e-11)
    assert (native-exact).abs().max()>1e-3
