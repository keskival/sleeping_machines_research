import copy
import unittest
import torch
from sleeping_machines.addressed_event_heads import AddressedEventHeads
from sleeping_machines.fast_native_core import fast_class
from sleeping_machines.packed_token_core import PackedTokenCore
from sleeping_machines.stage_gain_token_core import StageGainTokenCore
from sleeping_machines.sparse_counterfactual_episodes import token_features


def model(gains,tied=False):
    torch.manual_seed(7)
    m=fast_class(AddressedEventHeads)(sources=1,content_dim=7,classes=7,payload=4,depth=len(gains),heads=2,pool=3).double()
    for layer,gain in zip(m.units,gains):
        for head in layer:
            for pool in head:
                for unit in pool:unit.gain=gain
                if tied:
                    for unit in pool[1:]:
                        for name in ('input','output','gate','control'):setattr(unit,name,getattr(pool[0],name))
    return m


class StageGainTest(unittest.TestCase):
    def test_nonuniform_sparse_forward_state_routes_rng_and_every_gradient(self):
        ids=torch.tensor([[0,1,2],[3,4,5]])
        for tied in (False,True):
            old=model((.35,.1,0.),tied);packed=StageGainTokenCore(old)
            self.assertEqual([u[0][0][0].gain for u in old.units],[.35,.1,0.])
            a_rng=torch.Generator().manual_seed(11);b_rng=torch.Generator().manual_seed(11)
            a,sa,_=token_features(old,ids,generator=a_rng,route_credit='none')
            b,sb,_=token_features(packed,ids,generator=b_rng,route_credit='none')
            self.assertTrue(torch.equal(a,b));self.assertTrue(torch.equal(a_rng.get_state(),b_rng.get_state()))
            for key in sa:
                va,vb=sa[key],sb[key]
                if isinstance(va,list):self.assertTrue(all(torch.equal(x,y) for x,y in zip(va,vb)),key)
                else:self.assertTrue(torch.equal(va,vb),key)
            old.head(a).square().sum().backward();packed.head(b).square().sum().backward()
            for name,p in old.named_parameters():
                if p.grad is not None:self.assertTrue(torch.allclose(p.grad,packed.original_gradient(name),atol=1e-10,rtol=1e-10),name)

    def test_uniform_legacy_parity_and_checkpoint_restores_gains(self):
        old=model((.35,.35));legacy=PackedTokenCore(old);new=StageGainTokenCore(old)
        self.assertEqual(sum(p.numel() for p in legacy.parameters()),sum(p.numel() for p in new.parameters()))
        for a,b in zip(legacy._stacked(0),new._stacked(0)):
            for key in a:
                if isinstance(a[key],torch.Tensor):self.assertTrue(torch.equal(a[key],b[key]),key)
                else:self.assertEqual(a[key],b[key])
        saved=copy.deepcopy(StageGainTokenCore(model((.35,.1))).state_dict())
        new.load_state_dict(saved);self.assertEqual(new.stage_gains,(.35,.1))
        new.float();self.assertEqual(new.stage_gains,(.35,.1))
        bad=copy.deepcopy(saved);bad['_extra_state']['stage_gains']=(.1,)
        with self.assertRaises(ValueError):new.load_state_dict(bad)

    def test_completed_packed_conversion_preserves_rng_state_and_gradients(self):
        old=PackedTokenCore(model((.35,.35)))
        before=torch.get_rng_state().clone()
        grown=StageGainTokenCore.from_packed(old)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        for name,p in old.named_parameters():
            other=dict(grown.named_parameters())[name]
            self.assertTrue(torch.equal(p,other),name)
            self.assertNotEqual(p.data_ptr(),other.data_ptr(),name)
        ids=torch.tensor([[0,1,2],[3,4,5]])
        a,sa,_=token_features(old,ids,generator=torch.Generator().manual_seed(11),route_credit='none')
        b,sb,_=token_features(grown,ids,generator=torch.Generator().manual_seed(11),route_credit='none')
        self.assertTrue(torch.equal(a,b))
        for key in sa:
            if isinstance(sa[key],list):self.assertTrue(all(torch.equal(x,y) for x,y in zip(sa[key],sb[key])),key)
            else:self.assertTrue(torch.equal(sa[key],sb[key]),key)
        old.head(a).square().sum().backward();grown.head(b).square().sum().backward()
        for name,p in old.named_parameters():
            other=dict(grown.named_parameters())[name]
            if p.grad is not None:self.assertTrue(torch.equal(p.grad,other.grad),name)
        restored=StageGainTokenCore.from_packed(old)
        restored.load_state_dict(grown.state_dict())
        self.assertEqual(restored.stage_gains,grown.stage_gains)
        with self.assertRaises(ValueError):StageGainTokenCore.from_packed(grown)
