import torch
from sleeping_machines.prior_centered_readout import fit_prior_centered,linear_parameters


def test_positive_association_can_beat_the_constant_label_predictor():
    logits=torch.tensor([[.1,0.],[0.,.1],[.2,0.],[0.,.2]],dtype=torch.double)
    labels=torch.tensor([0,1,0,1]);fit=fit_prior_centered(logits,labels)
    assert fit['association']>0 and fit['gain']>0 and fit['nll']<fit['prior_nll']


def test_negative_association_returns_the_constant_predictor():
    z=torch.tensor([[0.,1.],[1.,0.]],dtype=torch.double);fit=fit_prior_centered(z,torch.tensor([0,1]))
    assert fit['gain']==0 and abs(fit['nll']-fit['prior_nll'])<1e-12


def test_class_bias_cannot_change_association_or_centred_calibration():
    z=torch.tensor([[.1,0.],[0.,.1],[.2,0.],[0.,.2]],dtype=torch.double);labels=torch.tensor([0,1,0,1])
    a=fit_prior_centered(z,labels);b=fit_prior_centered(z+torch.tensor([7.,-3.]),labels)
    assert abs(a['association']-b['association'])<1e-12
    assert abs(a['nll']-b['nll'])<1e-12


def test_calibration_is_a_linear_head_change_only():
    torch.manual_seed(19);linear=torch.nn.Linear(3,2).double();x=torch.randn(8,3,dtype=torch.double)
    labels=torch.tensor([0,1,0,1,0,1,0,1]);z=linear(x);fit=fit_prior_centered(z,labels)
    w,b=linear_parameters(linear.weight,linear.bias,fit)
    expected=fit['log_prior']+fit['gain']*(z-fit['center'])
    assert torch.allclose(torch.nn.functional.linear(x,w,b),expected,atol=1e-12,rtol=1e-12)
