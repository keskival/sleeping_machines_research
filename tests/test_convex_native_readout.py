import pytest
import torch
from sleeping_machines.convex_native_readout import fit_binary_readout,binary_logits


def test_convex_fit_solves_train_stationarity_and_affine_head_equivalence():
    torch.manual_seed(61);x=torch.randn(40,5,dtype=torch.float64)
    y=(x[:,0]-x[:,1]>0).long();fit=fit_binary_readout(x,y)
    z=binary_logits(x,fit)
    assert fit['max_gradient']<1e-8
    assert float(torch.nn.functional.cross_entropy(z,y))<.3
    head=torch.nn.Linear(5,2).double()
    with torch.no_grad():
        head.weight.zero_();head.bias.zero_()
        head.weight[1].copy_(fit['weight']);head.bias[1].copy_(fit['bias'])
    assert torch.allclose(head(x),z,atol=1e-12,rtol=1e-12)


def test_constant_features_return_balanced_prior():
    x=torch.ones(12,4);y=torch.tensor([0,1]*6)
    z=binary_logits(x,fit_binary_readout(x,y))
    assert torch.equal(z,torch.zeros_like(z))


def test_requires_train_support_and_fixed_positive_penalty():
    with pytest.raises(ValueError):fit_binary_readout(torch.ones(4,2),torch.zeros(4))
    with pytest.raises(ValueError):fit_binary_readout(torch.ones(4,2),torch.tensor([0,1]*2),ridge=0)
