"""Contracts of the competing-risks race readout and posterior routing (sleeping_machines/race_readout.py; §§432–433)."""
import numpy as np
import torch

from sleeping_machines.addressed_event_heads import AddressedEventHeads
from sleeping_machines.carried_episodes import carried_logits
from sleeping_machines.compiled_episodes import layer_step
from sleeping_machines.fast_native_core import fast_class
from sleeping_machines.race_readout import (RaceReadout, detach_state, event_terms, readout_episode, routed_layer_step)


def _model():
    torch.manual_seed(3)
    m = fast_class(AddressedEventHeads)(sources=1, content_dim=5, classes=7, payload=8, depth=2, heads=2, pool=3)
    r = RaceReadout(5, 2, 3, 8, 16, hidden=16, mu0=0.)
    return m, r


def _data(n=3, T=12, seed=0):
    rng = np.random.default_rng(seed)
    t = np.cumsum(rng.exponential(1.5, (n, T)), 1)
    ids = rng.integers(0, 5, (n, T))
    return torch.from_numpy(t), torch.from_numpy(np.eye(5, dtype=np.float32)[ids]), torch.from_numpy(ids)


def test_routed_step_without_override_is_layer_step():
    model, _ = _model(); stamps, marks, _ = _data()
    with torch.no_grad():
        a, _ = carried_logits(model, stamps, marks, step=layer_step, deterministic=True)
        b, _ = carried_logits(model, stamps, marks, step=lambda *x: routed_layer_step(*x[:24], x[24]), deterministic=True)
    torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_density_integrates_to_event_probability():
    torch.manual_seed(1)
    S, V = 4, 5
    o = torch.randn(1, S, V + 3, dtype=torch.float64)
    p = dict(logp=torch.log_softmax(o[..., :V], -1), mu=o[..., V], log_sigma=o[..., V + 1].clamp(-1, 1) * .5,
             logq=torch.nn.functional.logsigmoid(o[..., V + 2]), log1mq=torch.nn.functional.logsigmoid(-o[..., V + 2]))
    t_ref = torch.tensor([[0., -.5, -2., -.1]], dtype=torch.float64); t_now = torch.tensor([0.], dtype=torch.float64)
    grid = torch.exp(torch.linspace(-9, 6, 40001, dtype=torch.float64))        # t' - t_now on a log grid
    mass = 0.
    for e in range(V):
        ll, _, _ = event_terms({k: v.expand(len(grid), *v.shape[1:]) for k, v in p.items()}, t_ref.expand(len(grid), S),
                               t_now.expand(len(grid)), grid, torch.full((len(grid),), e), 1e-3)
        dens = ll.exp(); mass += float(torch.trapezoid(dens, grid))
    from sleeping_machines.race_readout import log_surv
    never = float((log_surv(p, torch.full((1, S), 1e9, dtype=torch.float64), 1e-3)
                   - log_surv(p, (t_now[:, None] - t_ref), 1e-3)).sum().exp())
    assert abs(mass + never - 1) < 2e-3, (mass, never)


def test_segments_chain_exactly():
    model, ro = _model(); stamps, marks, ids = _data()
    with torch.no_grad():
        a, at, av, _ = readout_episode(model, ro, stamps, marks, ids, deterministic=True)
        b1, bt1, bv1, st = readout_episode(model, ro, stamps[:, :5], marks[:, :5], ids[:, :5], deterministic=True)
        b2, bt2, bv2, _ = readout_episode(model, ro, stamps[:, 5:], marks[:, 5:], ids[:, 5:], state=detach_state(st),
                                          deterministic=True)
    torch.testing.assert_close(torch.cat([b1, b2], 1), a, rtol=1e-6, atol=1e-8)
    torch.testing.assert_close(torch.cat([bt1, bt2], 1), at, rtol=1e-6, atol=1e-8)
    assert not av[:, 0].any() and av[:, 1:].all() and bv2.all()


def test_posterior_routing_writes_the_most_responsible_slot():
    model, ro = _model(); stamps, marks, ids = _data(n=2, T=6)
    with torch.no_grad():
        _, _, _, st = readout_episode(model, ro, stamps[:, :5], marks[:, :5], ids[:, :5], posterior=True, deterministic=True)
        _, _, logr = event_terms(st['laws'], st['t_ref'], st['t_prev'], stamps[:, 5], ids[:, 5], ro.eps)
        best = logr.view(2, 2, 3).argmax(-1)
        _, _, _, st2 = readout_episode(model, ro, stamps[:, 5:], marks[:, 5:], ids[:, 5:], state=detach_state(st),
                                       posterior=True, deterministic=True)
    D = model.depth
    written = st2['arr'][D - 1] != st['arr'][D - 1]
    assert torch.equal(written, torch.nn.functional.one_hot(best, 3).bool())


def test_readout_gradients_reach_memory_path():
    model, ro = _model(); stamps, marks, ids = _data()
    ll, _, valid, _ = readout_episode(model, ro, stamps, marks, ids, deterministic=True)
    (-(ll * valid).sum()).backward()
    assert ro.out.weight.grad.abs().sum() > 0
    assert model.units[-1][0][0][0].input.weight.grad is not None or any(
        p.grad is not None and p.grad.abs().sum() > 0 for n, p in model.named_parameters() if 'input' in n)


def test_compiled_episode_equals_eager():
    model, ro = _model(); stamps, marks, ids = _data()
    with torch.no_grad():
        a = readout_episode(model, ro, stamps, marks, ids, posterior=True, deterministic=True)
        b = readout_episode(model, ro, stamps, marks, ids, posterior=True, deterministic=True, compiled=True)
    torch.testing.assert_close(a[0], b[0], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(a[1], b[1], rtol=1e-5, atol=1e-6)
