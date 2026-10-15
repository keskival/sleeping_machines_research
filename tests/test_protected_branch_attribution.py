import copy
import unittest
import torch
from sleeping_machines.addressed_event_heads import AddressedEventHeads
from sleeping_machines.fast_native_core import fast_class
from sleeping_machines.packed_token_core import PackedTokenCore
from sleeping_machines.optional_temporal_depth import grow_native_core
from sleeping_machines.protected_temporal_branch import protected_branch_features
from sleeping_machines.protected_branch_attribution import without_branch_memory, risk_decomposition


def core(depth):
    torch.manual_seed(13+depth)
    raw=fast_class(AddressedEventHeads)(sources=1,content_dim=7,classes=7,payload=4,depth=depth,heads=2,pool=3).double()
    return PackedTokenCore(raw)


class AttributionTest(unittest.TestCase):
    def test_erasure_preserves_parent_features_memory_context_arrivals_and_noise(self):
        first=grow_native_core(core(2),core(4));second=copy.deepcopy(first)
        ids=torch.tensor([[0,1,2,3],[3,4,5,6]])
        a=torch.Generator().manual_seed(9);b=torch.Generator().manual_seed(9)
        kwargs=dict(route_credit='none',parent_readout='all',pair=True,output='parent')
        x,s,_=protected_branch_features(first,ids,generator=a,**kwargs)
        y,t,_=without_branch_memory(protected_branch_features)(second,ids,generator=b,**kwargs)
        width=first.parent_depth*first.total_payload
        self.assertTrue(torch.equal(x[...,:width],y[...,:width]))
        self.assertGreater(float((x[...,width:]-y[...,width:]).detach().abs().max()),0.)
        for name in ('mem','arr','seen'):
            self.assertTrue(all(torch.equal(v,w) for v,w in zip(s[name][:2],t[name][:2])),name)
        for name in ('ctx_vals','ctx_arr','has_ctx','position','query_arrivals'):
            self.assertTrue(torch.equal(s[name],t[name]),name)
        self.assertTrue(torch.equal(a.get_state(),b.get_state()))
        self.assertTrue(torch.equal(first.new_depth_generator.get_state(),second.new_depth_generator.get_state()))

    def test_decomposition_and_invalid_inputs(self):
        r=risk_decomposition(6.,6.1,6.2)
        self.assertAlmostEqual(r['total_excess'],.1)
        self.assertAlmostEqual(r['correction_effect'],-.1)
        self.assertAlmostEqual(r['fitted_parent_path_excess'],.2)
        with self.assertRaises(ValueError):risk_decomposition(float('nan'),6.,6.)
        with self.assertRaises(ValueError):
            without_branch_memory(protected_branch_features)(None,None,route_credit='sampled')

if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
