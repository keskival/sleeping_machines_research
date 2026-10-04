import numpy as np
import torch

from sleeping_machines.addressed_event_heads import AddressedEventHeads
from sleeping_machines.batched_episodes import batched_logits
from sleeping_machines.fast_native_core import fast_class
from sleeping_machines.sparse_inference import sparse_logits


def test_winner_only_inference_equals_batched_evaluation():
    for pool, depth in ((1, 2), (3, 3), (4, 2)):
        torch.manual_seed(pool)
        m = fast_class(AddressedEventHeads)(sources=1, content_dim=5, classes=5, payload=8, depth=depth, heads=2,
                                            pool=pool).double().eval()
        with torch.no_grad():
            for p in m.parameters():
                p.add_(torch.randn_like(p) * .1)
        g = np.random.default_rng(pool)
        rows = [dict(events=[(float(t) + g.random(), g.normal(size=5)) for t in range(L)]) for L in (7, 3, 9, 1, 9)]
        with torch.no_grad():
            ref = batched_logits(m, rows, 11, all_logits=True)
        got = sparse_logits(m, rows, 11, all_logits=True)
        assert torch.allclose(got, ref, rtol=1e-10, atol=1e-10), (pool, float((got - ref).abs().max()))
        assert torch.allclose(sparse_logits(m, rows, 12), batched_logits(m, rows, 12).detach(), rtol=1e-10, atol=1e-10)


def test_deterministic_routing_matches_across_paths():
    from sleeping_machines.compiled_episodes import compiled_logits, layer_step
    from sleeping_machines.sparse_inference import SparseStepper
    torch.manual_seed(2)
    m = fast_class(AddressedEventHeads)(sources=1, content_dim=5, classes=5, payload=8, depth=2, heads=2, pool=3).double().eval()
    g = np.random.default_rng(3)
    rows = [dict(events=[(float(t), g.normal(size=5)) for t in range(6)])]
    with torch.no_grad():
        a = batched_logits(m, rows, 1, all_logits=True, deterministic=True)
        b = compiled_logits(m, rows, 99, all_logits=True, step=layer_step, deterministic=True)   # seed irrelevant
        st = SparseStepper(m, 1, 5, deterministic=True)
        c = torch.stack([st.step(torch.tensor([float(t)]), torch.tensor(rows[0]['events'][t][1])[None])[0] for t in range(6)])
    assert torch.allclose(a[0], b[0], atol=1e-10) and torch.allclose(a[0], c, atol=1e-10)
