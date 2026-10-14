"""Finite analytical witnesses; no benchmark fit or performance claim."""
import copy
import math
import unittest
import torch
from sleeping_machines.packed_token_core import PackedTokenCore
from sleeping_machines.sparse_counterfactual_episodes import token_features
from tests.test_stage_gain_token_core import model


def rotation():
    angles=torch.tensor([.37,-.61],dtype=torch.float64)
    r=torch.zeros(4,4,dtype=torch.float64)
    for i,angle in enumerate(angles):
        c,s=angle.cos(),angle.sin();j=2*i
        r[j:j+2,j:j+2]=torch.stack([torch.stack([c,-s]),torch.stack([s,c])])
    return r


def gauge(core,r):
    with torch.no_grad():
        core.banks['input'].copy_(r@core.banks['input'])
        for field in ('output','key_read'):core.banks[field].copy_(core.banks[field]@r.T)


def evaluate(core):
    return token_features(core,torch.tensor([[0,1,2,4],[3,4,5,0]]),
        generator=torch.Generator().manual_seed(11),route_credit='none')


class TemporalGeometryTest(unittest.TestCase):
    def test_modal_gauge_preserves_native_features_routes_clocks_and_memory_coordinates(self):
        old=PackedTokenCore(model((.35,.35)));new=copy.deepcopy(old);r=rotation();gauge(new,r)
        a,sa,pa=evaluate(old);b,sb,pb=evaluate(new)
        self.assertTrue(torch.allclose(a,b,atol=1e-12,rtol=1e-12))
        for key in sa:
            if key=='mem':
                for x,y in zip(sa[key],sb[key]):self.assertTrue(torch.allclose(x@r.T,y,atol=1e-12,rtol=1e-12))
            elif isinstance(sa[key],list):
                for x,y in zip(sa[key],sb[key]):self.assertTrue(torch.allclose(x,y,atol=1e-12,rtol=1e-12),key)
            elif sa[key].dtype in (torch.bool,torch.long):self.assertTrue(torch.equal(sa[key],sb[key]),key)
            else:self.assertTrue(torch.allclose(sa[key],sb[key],atol=1e-12,rtol=1e-12),key)
        for (_,x),(_,y) in zip(pa,pb):self.assertTrue(torch.allclose(x,y,atol=1e-12,rtol=1e-12))

    def test_sgd_respects_modal_gauge_but_diagonal_adam_does_not(self):
        r=rotation()
        for law in ('sgd','adam'):
            old=PackedTokenCore(model((.35,.35)));new=copy.deepcopy(old);gauge(new,r)
            for core in (old,new):
                features,_,_=evaluate(core);core.head(features).square().mean().backward()
                opt=torch.optim.SGD(core.parameters(),lr=.01) if law=='sgd' else torch.optim.AdamW(core.parameters(),lr=.01,weight_decay=0.)
                opt.step()
            expected=copy.deepcopy(old);gauge(expected,r)
            error=max(float((a-b).abs().max().detach()) for a,b in zip(expected.parameters(),new.parameters()))
            if law=='sgd':self.assertLess(error,1e-12)
            else:self.assertGreater(error,1e-5)

    def test_exact_zero_output_deadlock_with_available_xor_information(self):
        z=torch.tensor([[-1.,-1.],[-1.,1.],[1.,-1.],[1.,1.]],dtype=torch.float64)
        product=z.prod(-1);target=(1+product)/2
        def risk(p):return torch.nn.functional.binary_cross_entropy_with_logits(p[0]*(z[:,0]+p[1]*product),target)
        p=torch.zeros(2,dtype=torch.float64,requires_grad=True)
        g=torch.autograd.grad(risk(p),p)[0];self.assertTrue(torch.equal(g,torch.zeros_like(g)))
        h=torch.autograd.functional.hessian(risk,p)
        self.assertTrue(torch.equal(h,torch.tensor([[.25,-.5],[-.5,0.]],dtype=torch.float64)))
        self.assertLess(float(torch.linalg.eigvalsh(h).min()),0.)
        epsilon=.2;p=torch.tensor([epsilon,0.],dtype=torch.float64,requires_grad=True)
        self.assertAlmostEqual(float((risk(p)-math.log(2)).detach()),math.log(math.cosh(epsilon/2)),places=14)
        self.assertAlmostEqual(float(torch.autograd.grad(risk(p),p)[0][1]),-epsilon/2,places=14)

    def test_event_sampled_cascade_has_repeated_pole_kernel(self):
        a=.8;m1=m2=0.
        for n in range(10):
            m1=a*m1+(1. if n==0 else 0.);m2=a*m2+m1
            self.assertAlmostEqual(m2,(n+1)*a**n,places=13)
