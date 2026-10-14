import unittest
import torch
from sleeping_machines.controlled_route_credit import controlled_route_credit
from sleeping_machines.paired_route_credit import paired_route_credit


class ControlledCreditTest(unittest.TestCase):
    def test_full_proposal_mean_matches_original_logit_gradient(self):
        logits=torch.tensor([[.2,-.4,.8,.3]],dtype=torch.float64,requires_grad=True)
        pi=logits.softmax(-1);winner=torch.tensor([2])
        q=torch.tensor([.2,.3,0.,.5],dtype=torch.float64)
        labels=torch.tensor([1.2,-.4,0.,2.],dtype=torch.float64)
        baseline=torch.tensor([[.7,-1.,5.,1.4]],dtype=torch.float64,requires_grad=True)
        raw=controlled=0.
        for j in (0,1,3):
            alt=torch.tensor([j]);proposal=q[j].reshape(1);label=labels[j].reshape(1)
            raw=raw+q[j]*paired_route_credit(pi,alt,proposal,label)
            controlled=controlled+q[j]*controlled_route_credit(pi,winner,alt,proposal,label,baseline)
        g0=torch.autograd.grad(raw,logits,retain_graph=True)[0]
        g1,gb=torch.autograd.grad(controlled,(logits,baseline),allow_unused=True)
        self.assertTrue(torch.allclose(g0,g1,atol=1e-14,rtol=1e-14))
        self.assertIsNone(gb)
        self.assertEqual(float(controlled.detach()),0.)

    def test_perfect_baseline_removes_candidate_sampling_variance(self):
        logits=torch.tensor([[.2,-.4,.8,.3]],dtype=torch.float64,requires_grad=True)
        pi=logits.softmax(-1);winner=torch.tensor([2]);labels=torch.tensor([[1.2,-.4,0.,2.]],dtype=torch.float64)
        gradients=[]
        for j,q in ((0,.2),(1,.3),(3,.5)):
            term=controlled_route_credit(pi,winner,torch.tensor([j]),torch.tensor([q]),labels[:,j],labels)
            gradients.append(torch.autograd.grad(term,logits,retain_graph=True)[0])
        self.assertTrue(all(torch.equal(gradients[0],g) for g in gradients[1:]))

    def test_zero_baseline_retains_original_sample_credit(self):
        logits=torch.tensor([[.2,-.4,.8]],dtype=torch.float64,requires_grad=True);pi=logits.softmax(-1)
        alt=torch.tensor([0]);q=torch.tensor([.4]);label=torch.tensor([.7])
        raw=paired_route_credit(pi,alt,q,label)
        controlled=controlled_route_credit(pi,torch.tensor([2]),alt,q,label,torch.zeros_like(pi))
        a=torch.autograd.grad(raw,logits,retain_graph=True)[0];b=torch.autograd.grad(controlled,logits)[0]
        self.assertTrue(torch.equal(a,b))
