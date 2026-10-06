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


def test_smc_eval_single_particle_matches_episode():
    import sys
    sys.path.insert(0, 'experiments/fas')
    from race_smc_eval import smc_log_z
    model, ro = _model(); stamps, marks, ids = _data()
    with torch.no_grad():
        ll, _, valid, _ = readout_episode(model, ro, stamps, marks, ids, posterior=True, deterministic=True)
        lz = smc_log_z(model, ro, stamps, marks, ids, 1, True, deterministic=True)
    torch.testing.assert_close(lz, torch.cumsum(ll * valid, 1), rtol=1e-6, atol=1e-8)


def test_smc_eval_particles_finite_and_resample():
    import sys
    sys.path.insert(0, 'experiments/fas')
    from race_smc_eval import smc_log_z
    model, ro = _model(); stamps, marks, ids = _data()
    with torch.no_grad():
        lz = smc_log_z(model, ro, stamps, marks, ids, 8, True)
    assert torch.isfinite(lz).all() and lz.shape == stamps.shape


def test_type_durations_density_integrates_and_reduces_to_shared_law():
    torch.manual_seed(2)
    S, V = 3, 4
    o = torch.randn(1, S, 3 * V + 1, dtype=torch.float64)
    p = dict(logp=torch.log_softmax(o[..., :V], -1), mu=o[..., V:2 * V], log_sigma=o[..., 2 * V:3 * V].clamp(-1, 1) * .5,
             logq=torch.nn.functional.logsigmoid(o[..., -1]), log1mq=torch.nn.functional.logsigmoid(-o[..., -1]))
    t_ref = torch.tensor([[0., -.5, -2.]], dtype=torch.float64); t_now = torch.tensor([0.], dtype=torch.float64)
    grid = torch.exp(torch.linspace(-9, 6, 40001, dtype=torch.float64))
    from sleeping_machines.race_readout import log_surv
    mass = 0.
    for e in range(V):
        ll, _, _ = event_terms({k: v.expand(len(grid), *v.shape[1:]) for k, v in p.items()}, t_ref.expand(len(grid), S),
                               t_now.expand(len(grid)), grid, torch.full((len(grid),), e), 1e-3)
        mass += float(torch.trapezoid(ll.exp(), grid))
    never = float((log_surv(p, torch.full((1, S), 1e9, dtype=torch.float64), 1e-3)
                   - log_surv(p, (t_now[:, None] - t_ref), 1e-3)).sum().exp())
    assert abs(mass + never - 1) < 2e-3, (mass, never)
    # identical per-type laws reduce to the shared-law readout
    shared = dict(p, mu=p['mu'][..., 0], log_sigma=p['log_sigma'][..., 0])
    tied = dict(p, mu=p['mu'][..., :1].expand(-1, -1, V), log_sigma=p['log_sigma'][..., :1].expand(-1, -1, V))
    a = event_terms(shared, t_ref, t_now, torch.tensor([1.3], dtype=torch.float64), torch.tensor([2]), 1e-3)
    b = event_terms(tied, t_ref, t_now, torch.tensor([1.3], dtype=torch.float64), torch.tensor([2]), 1e-3)
    for x, y in zip(a, b):
        torch.testing.assert_close(x, y)


def test_type_durations_episode_runs_and_chains():
    torch.manual_seed(3)
    model, _ = _model(); ro = RaceReadout(5, 2, 3, 8, 16, hidden=16, mu0=0., type_durations=True)
    stamps, marks, ids = _data()
    with torch.no_grad():
        a = readout_episode(model, ro, stamps, marks, ids, posterior=True, deterministic=True)
        b1 = readout_episode(model, ro, stamps[:, :5], marks[:, :5], ids[:, :5], posterior=True, deterministic=True)
        b2 = readout_episode(model, ro, stamps[:, 5:], marks[:, 5:], ids[:, 5:], state=detach_state(b1[3]),
                             posterior=True, deterministic=True)
    torch.testing.assert_close(torch.cat([b1[0], b2[0]], 1), a[0], rtol=1e-6, atol=1e-8)


def _binding(Ub=5):
    from sleeping_machines.race_readout import BindingMemory
    torch.manual_seed(4)
    model, _ = _model()
    ro = RaceReadout(5, 1, Ub, 8, 16, hidden=16, mu0=0., type_durations=True)
    return model, ro, BindingMemory(16, Ub, 8, tau_max=100.)


def test_binding_segments_chain_exactly():
    model, ro, bm = _binding(); stamps, marks, ids = _data()
    with torch.no_grad():
        a = readout_episode(model, ro, stamps, marks, ids, deterministic=True, binding=bm)
        b1 = readout_episode(model, ro, stamps[:, :5], marks[:, :5], ids[:, :5], deterministic=True, binding=bm)
        b2 = readout_episode(model, ro, stamps[:, 5:], marks[:, 5:], ids[:, 5:], state=detach_state(b1[3]),
                             deterministic=True, binding=bm)
    torch.testing.assert_close(torch.cat([b1[0], b2[0]], 1), a[0], rtol=1e-6, atol=1e-8)
    assert a[3]['bseen'][:, 0].all()


def test_binding_writes_the_most_responsible_slot_and_learns():
    model, ro, bm = _binding(); stamps, marks, ids = _data(n=2, T=7)
    with torch.no_grad():
        st = readout_episode(model, ro, stamps[:, :6], marks[:, :6], ids[:, :6], deterministic=True, binding=bm)[3]
        _, _, logr = event_terms(st['laws'], st['t_ref'], st['t_prev'], stamps[:, 6], ids[:, 6], ro.eps)
        st2 = readout_episode(model, ro, stamps[:, 6:], marks[:, 6:], ids[:, 6:], state=detach_state(st),
                              deterministic=True, binding=bm)[3]
    changed = (st2['bmem'] != st['bmem']).any(-1)
    assert torch.equal(changed, torch.nn.functional.one_hot(logr.argmax(-1), 5).bool())
    ll, _, valid, _ = readout_episode(model, ro, stamps, marks, ids, binding=bm)
    (-(ll * valid).sum()).backward()
    assert bm.inp.weight.grad.abs().sum() > 0 and bm.raw_rate.grad is not None


def test_step_class_mixture_density_integrates_and_runs():
    torch.manual_seed(5)
    S, V, M = 3, 4, 2
    ro = RaceReadout(V, 1, S, 8, 16, hidden=8, mu0=0., classes=M)
    with torch.no_grad():
        p = ro(torch.randn(1, 1, S, 8), torch.randn(1, 16))
    t_ref = torch.tensor([[0., -.5, -2.]], dtype=torch.float64); t_now = torch.tensor([0.], dtype=torch.float64)
    grid = torch.exp(torch.linspace(-9, 7, 40001, dtype=torch.float64))
    from sleeping_machines.race_readout import log_surv
    mass = 0.
    for e in range(V):
        ll, _, _ = event_terms({k: v.expand(len(grid), *v.shape[1:]) for k, v in p.items()}, t_ref.expand(len(grid), S),
                               t_now.expand(len(grid)), grid, torch.full((len(grid),), e), 1e-3)
        mass += float(torch.trapezoid(ll.exp(), grid))
    never = float((log_surv(p, torch.full((1, S), 1e9, dtype=torch.float64), 1e-3)
                   - log_surv(p, (t_now[:, None] - t_ref), 1e-3)).sum().exp())
    assert abs(mass + never - 1) < 3e-3, (mass, never)
    from sleeping_machines.race_readout import BindingMemory
    model, _ = _model(); stamps, marks, ids = _data()
    ro5 = RaceReadout(5, 1, 4, 8, 16, hidden=8, mu0=0., classes=3); bm = BindingMemory(16, 4, 8)
    ll, _, valid, _ = readout_episode(model, ro5, stamps, marks, ids, binding=bm)
    assert torch.isfinite(ll).all()


def test_binding_compiled_equals_eager():
    from sleeping_machines.race_readout import BindingMemory
    torch.manual_seed(6)
    model, _ = _model(); stamps, marks, ids = _data()
    ro = RaceReadout(5, 1, 4, 8, 16, hidden=8, mu0=0., classes=2); bm = BindingMemory(16, 4, 8)
    with torch.no_grad():
        a = readout_episode(model, ro, stamps, marks, ids, deterministic=True, binding=bm)
        b = readout_episode(model, ro, stamps, marks, ids, deterministic=True, binding=bm, compiled=True)
    torch.testing.assert_close(a[0], b[0], rtol=1e-5, atol=1e-6)
