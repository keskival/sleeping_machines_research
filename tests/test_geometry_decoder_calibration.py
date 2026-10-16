import torch
from experiments.hierarchy.grounding_decoder_calibration import fit_affine_geometry,predict


def test_existing_linear_decoder_closed_form_stationarity():
    torch.manual_seed(1021);x=torch.randn(64,4,dtype=torch.float64);y=torch.randn(64,3,dtype=torch.float64)+10
    fit=fit_affine_geometry(x,y)
    assert fit['max_gradient']<1e-10
    assert (predict(x,fit)-y).square().mean() <= (y-y.mean(0)).square().mean()+1e-12


def test_centering_preserves_bias_only_decoder_for_constant_features():
    x=torch.ones(8,4,dtype=torch.float64);y=torch.arange(24,dtype=torch.float64).reshape(8,3)
    fit=fit_affine_geometry(x,y);assert torch.allclose(predict(x,fit),y.mean(0).expand_as(y))
