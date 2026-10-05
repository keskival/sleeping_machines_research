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


def test_recruit_layer_neutral_settings_equal_layer_step():
    model = _model(); stamps, marks = _data()
    with torch.no_grad():
        a, _ = carried_logits(model, stamps, marks, step=layer_step, deterministic=True)
        b, _, pis = carried_logits(model, stamps, marks, deterministic=True, recruit=dict(eager=True))
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert len(pis) == stamps.shape[1] * model.depth and pis[0][1].shape == (3, 2, 2)


def test_free_bias_prefers_unwritten_slots():
    model = _model(); stamps, marks = _data()
    with torch.no_grad():
        _, st0, _ = carried_logits(model, stamps, marks, deterministic=True, recruit=dict(eager=True))
        _, st1, _ = carried_logits(model, stamps, marks, deterministic=True, recruit=dict(eager=True, free_bias=30.))
    used0 = sum(int(s.sum()) for s in st0['seen']); used1 = sum(int(s.sum()) for s in st1['seen'])
    assert used1 >= used0 and used1 == sum(s.numel() for s in st1['seen'])   # a huge bonus fills every slot


def test_balance_penalty_range():
    from sleeping_machines.recruit_layer import balance_penalty
    uniform = [(0, torch.full((4, 2, 4), .25))]; collapsed = [(0, torch.tensor([1., 0, 0, 0]).expand(4, 2, 4))]
    assert abs(float(balance_penalty(uniform)) - 1) < 1e-6 and abs(float(balance_penalty(collapsed)) - 4) < 1e-6


def test_stale_bias_neutral_at_zero_and_fills_slots_when_large():
    model = _model(); stamps, marks = _data()
    with torch.no_grad():
        a, _, _ = carried_logits(model, stamps, marks, deterministic=True, recruit=dict(eager=True))
        b, _, _ = carried_logits(model, stamps, marks, deterministic=True, recruit=dict(eager=True, stale_bias=0.))
        _, st, _ = carried_logits(model, stamps, marks, deterministic=True, recruit=dict(eager=True, stale_bias=30.))
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert sum(int(s.sum()) for s in st['seen']) == sum(s.numel() for s in st['seen'])
