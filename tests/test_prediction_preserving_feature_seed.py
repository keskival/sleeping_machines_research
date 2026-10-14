import unittest
import torch
from sleeping_machines.prediction_preserving_feature_seed import cross_credit_direction,bounded_seed_direction
from experiments.market.b10_depth_tpp_v1 import IdentityGrowthLayer
from experiments.tpp.race_tpp_v5 import TemporalMemoryLayer


class FeatureSeedTest(unittest.TestCase):
    def test_xor_cross_credit_unlocks_learning_without_initial_prediction_cost(self):
        z=torch.tensor([[-1.,-1.],[-1.,1.],[1.,-1.],[1.,1.]],dtype=torch.float64)
        product=z.prod(-1);target=(1+product)/2
        output=torch.nn.Parameter(torch.tensor(0.,dtype=torch.float64))
        feature=torch.nn.Parameter(torch.tensor(0.,dtype=torch.float64))
        def risk():return torch.nn.functional.binary_cross_entropy_with_logits(output*(z[:,0]+feature*product),target)
        initial=risk().detach();rng=torch.get_rng_state().clone()
        direction=cross_credit_direction(risk(),[output],[feature],[torch.ones_like(output)])
        self.assertEqual(float(direction[0]),-.5)
        delta=bounded_seed_direction(direction,.1)
        with torch.no_grad():feature.add_(delta[0])
        self.assertTrue(torch.equal(initial,risk().detach()))
        gradient=torch.autograd.grad(risk(),output)[0]
        self.assertGreater(abs(float(gradient)),0.)
        with torch.no_grad():output.add_(-.1*gradient)
        self.assertLess(float(risk().detach()),float(initial))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))

    def test_temporal_memory_feature_seed_preserves_the_contained_function(self):
        torch.manual_seed(7)
        branch=IdentityGrowthLayer(TemporalMemoryLayer(4,2,0.),4).double().eval()
        x=torch.tensor([[[.2,.4,-.1,.3],[.5,-.2,.3,.1],[.1,.8,-.4,.2]]],dtype=torch.float64)
        dt=torch.tensor([[0.,.1,.7]],dtype=torch.float64)
        target=x.roll(1,-1)
        parameters=(branch.temporal.write.weight,branch.temporal.read.weight,branch.temporal.freq,branch.temporal.log_rate)
        loss=(branch(x,dt)-target).square().mean()
        output_gradient=torch.autograd.grad(loss,branch.correction.weight,create_graph=True,retain_graph=True)[0]
        probe=output_gradient.detach()
        directions=cross_credit_direction(loss,[branch.correction.weight],parameters,[probe])
        self.assertGreater(sum(float(d.square().sum()) for d in directions),0.)
        before=branch(x,dt).detach();delta=bounded_seed_direction(directions,1e-4)
        with torch.no_grad():
            for p,d in zip(parameters,delta):p.add_(d)
        self.assertTrue(torch.equal(before,branch(x,dt).detach()))
        after=(branch(x,dt)-target).square().mean()
        new_gradient=torch.autograd.grad(after,branch.correction.weight)[0]
        self.assertGreater(float(new_gradient.square().sum()),float(probe.square().sum()))

    def test_nonzero_outputs_and_aliased_groups_are_rejected(self):
        a=torch.nn.Parameter(torch.tensor(1.));b=torch.nn.Parameter(torch.tensor(0.))
        with self.assertRaises(ValueError):cross_credit_direction(a*b,[a],[b],[torch.ones_like(a)])
        with torch.no_grad():a.zero_()
        with self.assertRaises(ValueError):cross_credit_direction(a*a,[a],[a],[torch.ones_like(a)])

    def test_equal_weight_marginals_do_not_match_composed_function(self):
        b=torch.tensor([-1.,1.])
        positive=b.clone();negative=-b
        self.assertTrue(torch.equal(positive.sort().values,negative.sort().values))
        self.assertEqual(float((b*positive).mean()),1.)
        self.assertEqual(float((b*negative).mean()),-1.)
