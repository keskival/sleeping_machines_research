import numpy as np
import torch
from experiments.r1_local_native_deep import LocalNativeDeepLM,word_classes
from sleeping_machines.document_boundary_causes import document_cause_law,enable_document_relevance,original_cause_law


def setup():
    torch.manual_seed(16701);c,w=word_classes(np.arange(1,13),3)
    model=LocalNativeDeepLM(12,c,w,width=8,depth=3,heads=2,pool=3,concept_units=16,topk=2,explore=0,dk=4,packed=True).double()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001);views=torch.randn(3,8,dtype=torch.float64)
    candidates=dict(features=torch.randn(4,3,8,dtype=torch.float64),tokens=torch.tensor([1,2,3,4]),successors=torch.tensor([2,3,4,5]),ages=torch.tensor([1.,2.,3.,4.],dtype=torch.float64),valid=torch.ones(4,dtype=torch.bool),suffix_matches=torch.tensor([0,1,2,0]),same_document=torch.tensor([True,False,True,False]))
    return model,optimizer,views,candidates


def test_zero_bias_preserves_predictions_and_existing_gradients():
    model,opt,h,c=setup();before=original_cause_law(model.causes,h,3,c)[0]
    before.sum().backward();grads={n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True);enable_document_relevance(model,opt)
    after=document_cause_law(model.causes,h,3,c)[0];after.sum().backward()
    assert torch.equal(before,after)
    for n,p in model.named_parameters():
        if n in grads:assert torch.allclose(p.grad,grads[n],atol=2e-12,rtol=2e-12),n
    assert model.causes.document_bias.grad is not None and model.causes.document_bias.grad.norm()>0


def test_nonzero_document_law_normalizes_and_gradient_matches_finite_difference():
    model,opt,h,c=setup();enable_document_relevance(model,opt)
    with torch.no_grad():model.causes.document_bias.copy_(torch.tensor([-1.,-.5],dtype=torch.float64))
    losses=torch.stack([document_cause_law(model.causes,h,y,c)[0] for y in range(12)])
    assert torch.allclose((-losses).exp().sum(0),torch.ones(3,dtype=torch.float64),atol=2e-12,rtol=2e-12)
    loss=document_cause_law(model.causes,h,3,c)[0].sum();expected=torch.autograd.grad(loss,model.causes.document_bias)[0]
    for j in range(2):
        with torch.no_grad():model.causes.document_bias[j]+=1e-5
        plus=document_cause_law(model.causes,h,3,c)[0].sum()
        with torch.no_grad():model.causes.document_bias[j]-=2e-5
        minus=document_cause_law(model.causes,h,3,c)[0].sum()
        with torch.no_grad():model.causes.document_bias[j]+=1e-5
        assert abs(float((plus-minus)/2e-5-expected[j]))<1e-8
