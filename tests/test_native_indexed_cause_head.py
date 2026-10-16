import numpy as np
import torch
from experiments.r1_local_native_deep import LocalNativeDeepLM,word_classes
from sleeping_machines.native_cause_bank import NativeCauseBank
from sleeping_machines.native_indexed_cause_head import indexed_cause_nll,shared_target_word_law


def model_and_bank():
    torch.manual_seed(115);c,w=word_classes(np.arange(1,14),4)
    model=LocalNativeDeepLM(13,c,w,width=8,depth=3,heads=2,pool=3,concept_units=16,topk=2,explore=1,dk=4)
    bank=NativeCauseBank(3,8,capacity=20)
    for t,token in enumerate([1,4,2,1]):bank.append(token,torch.randn(3,8),[('unit',t%2)],t)
    return model,bank


def test_indexed_cause_and_depth_laws_normalize_over_vocabulary():
    model,bank=model_and_bank();model.eval();views=torch.randn(3,8)
    candidates=bank.query(1,[('unit',0)],3)
    prior=torch.log_softmax(model.depth_query(views).squeeze(-1)+model.depth_bias,-1)
    masses=[];mix=[]
    for token in range(13):
        nll,_,_=indexed_cause_nll(model.causes,views,token,candidates)
        masses.append((-nll).exp());mix.append(torch.logsumexp(prior-nll,-1).exp())
    assert torch.allclose(torch.stack(masses).sum(0),torch.ones(3),atol=1e-6)
    assert torch.allclose(torch.stack(mix).sum(),torch.tensor(1.),atol=1e-6)


def test_indexed_successor_semantics_and_local_credit_are_live():
    model,bank=model_and_bank();views=torch.randn(3,8,requires_grad=True)
    candidates=bank.query(1,[('unit',0)],3)
    nll,aux,report=indexed_cause_nll(model.causes,views,4,candidates,explore=True)
    (nll.mean()+aux).backward()
    assert views.grad is not None and views.grad.norm()>0
    assert report['posterior_message'].shape==(3,8) and not report['posterior_message'].requires_grad
    assert torch.allclose(report['cause_responsibility'].sum(-1),torch.ones(3),atol=1e-6)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.causes.parameters())


def test_shared_actual_word_rows_match_existing_law_and_every_parameter_gradient():
    model,_=model_and_bank();head=model.causes.double();states=torch.randn(3,4,8,dtype=torch.float64,requires_grad=True)
    params=[states,head.class_emb,head.class_bias,head.embed.weight,head.word_bias]
    for token in [0,4,12]:
        existing=head.two_level(states,torch.full((3,),token,dtype=torch.long))
        shared=shared_target_word_law(head,states,token)
        assert torch.allclose(existing,shared,atol=1e-12,rtol=1e-12)
        eg=torch.autograd.grad(existing.sum(),params,retain_graph=True)
        sg=torch.autograd.grad(shared.sum(),params,retain_graph=True)
        for a,b in zip(eg,sg):assert torch.allclose(a,b,atol=1e-12,rtol=1e-12)
