import pytest
import torch
from sleeping_machines.sparse_race_language import TemporalUnit
from sleeping_machines.native_receiver_eligibility import ReceiverEligibility


@pytest.mark.parametrize('length',[1,7,41])
def test_actual_native_write_credit_matches_frozen_parameter_unroll(length):
    torch.manual_seed(710)
    unit=TemporalUnit(6,.2).double()
    with torch.no_grad():unit.control.weight.normal_(0,.2);unit.control.bias.normal_(0,.1)
    messages=torch.randn(length,6,dtype=torch.float64)
    gaps=torch.rand(length,dtype=torch.float64);gaps[::3]=0;times=gaps.cumsum(0)
    readout=torch.randn(6,3,dtype=torch.float64)
    memory=torch.zeros(6,dtype=torch.float64);previous=None;losses=[];states=[]
    for x,stamp in zip(messages,times):
        _,memory=unit.propose(x,memory,previous,stamp);previous=stamp
        losses.append(torch.sin(memory@readout).square().mean());states.append(memory.detach())
    names=['input.weight','control.weight','control.bias','raw_rate','frequency'];params=dict(unit.named_parameters())
    expected=torch.autograd.grad(torch.stack(losses).mean(),[params[n] for n in names],allow_unused=True)
    state=ReceiverEligibility.empty(unit);total={n:torch.zeros_like(params[n]) for n in names};bytes0=state.storage()['tensor_bytes']
    for j,(x,stamp) in enumerate(zip(messages,times)):
        state=state.propose(unit,x,stamp)
        assert torch.allclose(state.memory(),states[j],atol=2e-12,rtol=2e-12)
        leaf=state.memory().requires_grad_();loss=torch.sin(leaf@readout).square().mean()/length
        cotangent=torch.autograd.grad(loss,leaf)[0]
        for n,g in state.credit(cotangent).items():total[n]+=g
        assert state.storage()['tensor_bytes']==bytes0
        assert all(getattr(state,n).grad_fn is None for n in ['value','input_real','control_weight','frequency'])
    for n,g in zip(names,expected):
        g=torch.zeros_like(params[n]) if g is None else g
        assert torch.allclose(total[n],g,atol=2e-11,rtol=2e-11),n


def test_counterfactual_proposal_does_not_commit_receiver_state():
    unit=TemporalUnit(4,.2);old=ReceiverEligibility.empty(unit)
    proposal=old.propose(unit,torch.ones(4),10.)
    assert old.timestamp is None and old.memory().abs().sum()==0
    assert proposal.timestamp==10. and proposal.memory().abs().sum()>0
    with pytest.raises(ValueError):proposal.propose(unit,torch.ones(4),9.)
