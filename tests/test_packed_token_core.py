import torch
from test_token_episodes import model
from sleeping_machines.packed_token_core import PackedTokenCore
from sleeping_machines.token_episodes import token_features


def test_packed_and_module_cores_have_identical_features_and_every_leaf_gradient():
    for tied in (False,True):
        original=model()
        if tied:
            for layer in original.units:
                for head in layer:
                    for pool in head:
                        for unit in pool[1:]:
                            for name in ('input','output','gate','control'):
                                setattr(unit,name,getattr(pool[0],name))
        packed=PackedTokenCore(original)
        assert sum(p.numel() for p in original.parameters())==sum(p.numel() for p in packed.parameters())
        ids=torch.tensor([[0,1,2],[3,4,5]])
        a,sa,_=token_features(original,ids,generator=torch.Generator().manual_seed(19))
        b,sb,_=token_features(packed,ids,generator=torch.Generator().manual_seed(19))
        assert torch.equal(a,b)
        assert all(torch.equal(x,y) for x,y in zip(sa['mem'],sb['mem']))
        original.head(a).square().sum().backward();packed.head(b).square().sum().backward()
        for name,p in original.named_parameters():
            if p.grad is not None:
                assert torch.allclose(p.grad,packed.original_gradient(name),atol=1e-10,rtol=1e-10),name
