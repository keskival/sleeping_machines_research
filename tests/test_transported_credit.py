"""Transported write credit (sleeping_machines/transported_credit.py; THEORY §430): forward unchanged; with the write
term off, gradients equal the standard linear-credit path; with it on, only score-path gradients change."""
import torch

from sleeping_machines.carried_episodes import carried_logits
from sleeping_machines.compiled_episodes import layer_step
from sleeping_machines.transported_credit import transported_logits
from test_carried_episodes import _data, _model


def _grads(model, logits):
    model.zero_grad(); (logits ** 2).sum().backward()
    return [p.grad.clone() if p.grad is not None else torch.zeros_like(p) for p in model.parameters()]


def test_forward_equals_standard_layer():
    model = _model(); stamps, marks = _data()
    with torch.no_grad():
        a, _ = carried_logits(model, stamps, marks, step=layer_step, route_credit='linear', seed=3)
        b, _, rec = transported_logits(model, stamps, marks, seed=3)
    torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-7)
    assert len(rec) == stamps.shape[1] * model.depth


def test_write_term_off_equals_linear_credit_gradients():
    model = _model(); stamps, marks = _data()
    a, _ = carried_logits(model, stamps, marks, step=layer_step, route_credit='linear', seed=3)
    ga = _grads(model, a)
    b, _, _ = transported_logits(model, stamps, marks, seed=3, write_credit=False)
    gb = _grads(model, b)
    for x, y in zip(ga, gb):
        torch.testing.assert_close(x, y, rtol=1e-5, atol=1e-7)


def test_write_term_changes_gradients_but_not_values():
    model = _model(); stamps, marks = _data()
    a, _, _ = transported_logits(model, stamps, marks, seed=3, write_credit=False)
    b, _, _ = transported_logits(model, stamps, marks, seed=3, write_credit=True)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    ga, gb = _grads(model, a), _grads(model, b)
    assert any(not torch.allclose(x, y) for x, y in zip(ga, gb))
    assert all(torch.isfinite(y).all() for y in gb)
