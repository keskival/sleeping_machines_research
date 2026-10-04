import numpy as np
import torch

from sleeping_machines.addressed_event_heads import AddressedEventHeads
from sleeping_machines.compiled_episodes import compiled_logits
from sleeping_machines.expected_reception import ExpectedStepper, expected_logits
from sleeping_machines.fast_native_core import fast_class


def _model(pool, depth=2):
    torch.manual_seed(pool)
    m = fast_class(AddressedEventHeads)(sources=1, content_dim=4, classes=3, payload=8, depth=depth, heads=2,
                                        pool=pool).double()
    with torch.no_grad():
        for p in m.parameters():
            p.add_(torch.randn_like(p) * .1)
    return m


def _rows(L=8, lanes=3):
    g = np.random.default_rng(1)
    return [dict(events=[(float(t), g.normal(size=4)) for t in range(L)]) for _ in range(lanes)]


def test_stepper_equals_expected_logits_and_is_seed_independent():
    m = _model(3, 3).eval(); rows = _rows()
    with torch.no_grad():
        ref = expected_logits(m, rows, 5, all_logits=True, compiled=False)
        assert torch.equal(ref, expected_logits(m, rows, 99, all_logits=True, compiled=False))
        st = ExpectedStepper(m, len(rows))
        got = torch.stack([st.step(torch.tensor([r['events'][t][0] for r in rows]),
                                   torch.tensor(np.stack([r['events'][t][1] for r in rows]))) for t in range(8)], 1)
    assert torch.allclose(got, ref, atol=1e-12)


def test_pool_one_equals_deterministic_winner_race():
    m = _model(1).eval(); rows = _rows()
    with torch.no_grad():
        a = expected_logits(m, rows, all_logits=True, compiled=False)
        b = compiled_logits(m, rows, 3, all_logits=True, deterministic=True,
                            step=__import__('sleeping_machines.compiled_episodes', fromlist=['layer_step']).layer_step)
    assert torch.allclose(a, b, atol=1e-12)


def test_compiled_matches_eager_with_gradients():
    m = _model(3); rows = _rows()
    out = []
    for compiled in (False, True):
        m.zero_grad()
        z = expected_logits(m, rows, all_logits=True, compiled=compiled); z.square().sum().backward()
        out.append((z.detach().clone(), [p.grad.clone() if p.grad is not None else torch.zeros_like(p) for p in m.parameters()]))
    assert torch.allclose(out[0][0], out[1][0], atol=1e-9)
    for a, b in zip(out[0][1], out[1][1]):
        assert torch.allclose(a, b, rtol=1e-8, atol=1e-10)
