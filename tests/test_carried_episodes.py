"""Carried-state segments reproduce the full-sequence event loop (sleeping_machines/carried_episodes.py)."""
import numpy as np
import torch

from sleeping_machines.addressed_event_heads import AddressedEventHeads
from sleeping_machines.carried_episodes import carried_logits, detach
from sleeping_machines.compiled_episodes import compiled_logits, layer_step
from sleeping_machines.fast_native_core import fast_class


def _model():
    torch.manual_seed(3)
    return fast_class(AddressedEventHeads)(sources=1, content_dim=5, classes=7, payload=8, depth=2, heads=2, pool=2)


def _data(n=3, T=12, seed=0):
    rng = np.random.default_rng(seed)
    t = np.cumsum(rng.exponential(1.5, (n, T)), 1)
    ids = rng.integers(0, 5, (n, T))
    marks = torch.from_numpy(np.eye(5, dtype=np.float32)[ids])
    return torch.from_numpy(t), marks


def test_segments_with_carried_state_equal_full_sequence():
    model = _model(); stamps, marks = _data()
    rows = [dict(events=[(float(stamps[i, k]), marks[i, k]) for k in range(stamps.shape[1])]) for i in range(stamps.shape[0])]
    with torch.no_grad():
        full = compiled_logits(model, rows, 0, all_logits=True, step=layer_step, deterministic=True)
        a, st = carried_logits(model, stamps[:, :5], marks[:, :5], step=layer_step, deterministic=True)
        b, _ = carried_logits(model, stamps[:, 5:], marks[:, 5:], state=detach(st), step=layer_step, deterministic=True)
    torch.testing.assert_close(torch.cat([a, b], 1), full, rtol=1e-5, atol=1e-6)


def test_detached_state_truncates_credit_only():
    model = _model(); stamps, marks = _data()
    a, st = carried_logits(model, stamps[:, :5], marks[:, :5], step=layer_step, deterministic=True)
    b, _ = carried_logits(model, stamps[:, 5:], marks[:, 5:], state=detach(st), step=layer_step, deterministic=True)
    b.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert not any(x.requires_grad for x in detach(st)['mem'])
