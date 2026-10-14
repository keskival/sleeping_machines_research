import sys
import unittest
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/market'))
from b10_depth_tpp_v1 import Race, grow_from_parent
from b10_tpp import Race as ParentRace


class MarketDepthTest(unittest.TestCase):
    def test_two_layer_constructor_retains_completed_reference(self):
        gaps=np.array([.001,.01,.1])
        torch.manual_seed(7);parent=ParentRace(gaps,32,8)
        torch.manual_seed(7);candidate=Race(gaps,32,8,layers=2)
        self.assertEqual(set(parent.state_dict()),set(candidate.state_dict()))
        self.assertTrue(all(torch.equal(value,candidate.state_dict()[key]) for key,value in parent.state_dict().items()))
        parent.double().eval();candidate.double().eval()
        t=torch.tensor([[0.,0.,.001,.01]],dtype=torch.float64);m=torch.tensor([[0,2,1,3]])
        self.assertTrue(torch.equal(parent.terms(t,m),candidate.terms(t,m)))

    def test_eight_stages_receive_gradients_and_future_events_cannot_change_prefix(self):
        torch.manual_seed(7);model=Race(np.array([.001,.01,.1]),32,8,layers=8).double().eval()
        t=torch.tensor([[0.,0.,.001,.01]],dtype=torch.float64)
        m=torch.tensor([[0,2,1,3]])
        original=model.terms(t,m)
        changed_t=t.clone();changed_t[0,-1]=.2
        changed_m=m.clone();changed_m[0,-1]=4
        self.assertTrue(torch.equal(original[:,:2],model.terms(changed_t,changed_m)[:,:2]))
        (-original.mean()).backward()
        self.assertEqual(len(model.m.layers),8)
        for layer in model.m.layers:
            norm=sum(float(p.grad.square().sum()) for p in layer.parameters() if p.grad is not None)
            self.assertGreater(norm,0.)
            self.assertTrue(np.isfinite(norm))

    def test_growth_preserves_predictions_and_old_gradients_with_live_new_outputs(self):
        torch.manual_seed(7)
        parent=Race(np.array([.001,.01,.1]),32,8,layers=2).double().eval()
        grown=Race(np.array([.001,.01,.1]),32,8,layers=8).double().eval()
        grow_from_parent(grown,parent.state_dict())
        t=torch.tensor([[0.,0.,.001,.01]],dtype=torch.float64);m=torch.tensor([[0,2,1,3]])
        a=parent.terms(t,m);b=grown.terms(t,m)
        self.assertTrue(torch.equal(a,b))
        (-a.mean()).backward();(-b.mean()).backward()
        new=dict(grown.named_parameters())
        for name,p in parent.named_parameters():
            self.assertTrue(torch.equal(p.grad,new[name].grad),name)
        for layer in grown.m.layers[2:]:
            self.assertGreater(float(layer.correction.weight.grad.square().sum()),0.)
