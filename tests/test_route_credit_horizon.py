import unittest
import torch
from sleeping_machines.route_credit_horizon import credit_endpoint, HorizonDepthCreditAllocator
from sleeping_machines.depth_credit_allocator import DepthCreditAllocator


class HorizonTest(unittest.TestCase):
    def test_one_window_preserves_every_original_boundary(self):
        for length in (1, 17, 64):
            for token in range(length):
                self.assertEqual(credit_endpoint(token,length,16),min(length,(token//16+1)*16))

    def test_extended_horizon_reaches_next_detached_window_and_caps_at_chunk(self):
        self.assertEqual(credit_endpoint(15,64,16,4),64)
        self.assertEqual(credit_endpoint(48,55,16,4),55)
        with self.assertRaises(ValueError):credit_endpoint(64,64,16,4)

    def test_allocator_proxy_tracks_the_actual_suffix_weight(self):
        pi=torch.tensor([[.2,.3,.5]],dtype=torch.float64)
        alternative=torch.tensor([1]);q=torch.tensor([.5]);label=torch.tensor([2.])
        base=DepthCreditAllocator(8,1,16)
        extended=HorizonDepthCreditAllocator(8,1,16,consequence_windows=4)
        issued=dict(token=15,layer=0,head=0,length=64,probability=1/512,layer_probability=1/8)
        base.last=dict(issued);extended.last=dict(issued)
        base.observe(pi,alternative,q,label);extended.observe(pi,alternative,q,label)
        self.assertAlmostEqual(float(extended.second[0]/base.second[0]),49**2)

    def test_one_window_allocation_is_exactly_retained(self):
        base=DepthCreditAllocator(8,1,16)
        extended=HorizonDepthCreditAllocator(8,1,16)
        gen1=torch.Generator().manual_seed(17);gen2=torch.Generator().manual_seed(17)
        self.assertEqual(base.sample_site(64,8,1,gen1),extended.sample_site(64,8,1,gen2))
        args=(torch.tensor([[.2,.3,.5]],dtype=torch.float64),torch.tensor([1]),torch.tensor([.5]),torch.tensor([2.]))
        base.observe(*args);extended.observe(*args)
        self.assertEqual(base.packet(),extended.packet())
