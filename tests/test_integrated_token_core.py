"""Contracts for the sparse factual/local-credit/future-write construction."""
import torch
from test_token_episodes import model
from sleeping_machines.packed_token_core import PackedTokenCore
from sleeping_machines.sparse_token_episodes import token_features as parent
from sleeping_machines.sparse_counterfactual_episodes import token_features, detach
import test_counterfactual_token_episodes as future_contracts
from sleeping_machines.paired_route_credit import paired_route_credit


def test_packed_sparse_factual_and_local_credit_match_parent_after_parameter_change():
    original = model()
    ids = torch.tensor([[0, 1, 2], [3, 6, 4]])
    _, state, _ = parent(original, ids, generator=torch.Generator().manual_seed(3),
                         route_credit='none', eos=6)
    state = detach(state)
    with torch.no_grad():
        for p in original.parameters():
            p.add_(.013)
    packed = PackedTokenCore(original)
    runs = []
    for core, fn in ((original, parent), (packed, token_features)):
        x, st, _ = fn(core, ids, state, generator=torch.Generator().manual_seed(9),
                      alternative_generator=torch.Generator().manual_seed(31), eos=6)
        core.head(x).square().sum().backward()
        runs.append((x, st))
    assert torch.equal(runs[0][0], runs[1][0])
    for key in ('mem', 'arr', 'seen'):
        assert all(torch.equal(a, b) for a, b in zip(runs[0][1][key], runs[1][1][key]))
    for name, p in original.named_parameters():
        if p.grad is not None:
            assert torch.allclose(p.grad, packed.original_gradient(name), atol=1e-10, rtol=1e-10), name


def test_sparse_actual_write_has_future_effect_and_correct_conditional_credit(monkeypatch):
    # Reuse the equal-value/different-write and exact conditional-credit
    # witnesses with sparse execution. Packed parity is checked separately.
    def sparse_future(m, ids, *args, **kwargs):
        kwargs['route_credit'] = 'none'
        return token_features(m, ids, *args, **kwargs)
    monkeypatch.setattr(future_contracts, 'counterfactual_features', sparse_future)
    future_contracts.test_equal_immediate_deliveries_can_have_different_future_write_outcomes()
    future_contracts.test_enumerated_paired_suffix_teacher_equals_conditional_expected_loss_gradient()


def test_partition_preserves_sparse_and_future_route_state_with_packed_parameters():
    m = PackedTokenCore(model())
    ids = torch.tensor([[0, 1, 2, 3], [3, 4, 5, 6]])
    g = torch.Generator().manual_seed(8)
    ag = torch.Generator().manual_seed(12)
    a, sa, _ = token_features(m, ids, generator=g, alternative_generator=ag, eos=6)
    g2 = torch.Generator().manual_seed(8)
    ag2 = torch.Generator().manual_seed(12)
    b, sb, _ = token_features(m, ids[:, :2], generator=g2, alternative_generator=ag2, eos=6)
    c, sb, _ = token_features(m, ids[:, 2:], sb, generator=g2, alternative_generator=ag2, eos=6)
    assert torch.equal(a, torch.cat((b, c), 1))
    assert torch.equal(g.get_state(), g2.get_state())
    assert torch.equal(ag.get_state(), ag2.get_state())
    assert all(torch.equal(x, y) for x, y in zip(sa['mem'], sb['mem']))
    assert torch.equal(sa['race_winners'][2:], sb['race_winners'])


def test_actual_estimator_matches_expected_utility_and_detaches_replay_and_proposal():
    logits = torch.tensor([[.2, -.7, .6], [-.3, .8, .1]], dtype=torch.double, requires_grad=True)
    pi = logits.softmax(-1)
    winners = torch.tensor([0, 1])
    costs = torch.tensor([[2., 1., 4.], [3., 5., 1.]], dtype=torch.double, requires_grad=True)
    base = costs.gather(1, winners[:, None]).squeeze(1)
    eligible = 1 - torch.nn.functional.one_hot(winners, 3).double()
    conditional = pi * eligible
    conditional = conditional / conditional.sum(-1, keepdim=True)
    proposal = .9 * conditional + .1 * eligible / 2
    expected = 0.
    for offset in (1, 2):
        alt = (winners + offset) % 3
        q = proposal.gather(1, alt[:, None]).squeeze(1)
        delta = costs.gather(1, alt[:, None]).squeeze(1) - base
        term = paired_route_credit(pi, alt, q, delta)
        assert term.item() == 0.
        # Weight by the actual sampling law without differentiating sampling.
        expected = expected + paired_route_credit(pi, alt, q, delta * q.detach())
        assert torch.autograd.grad(term, costs, allow_unused=True, retain_graph=True)[0] is None
    exact = torch.autograd.grad((pi * costs.detach()).sum(-1).mean(), logits, retain_graph=True)[0]
    actual = torch.autograd.grad(expected, logits)[0]
    assert torch.allclose(exact, actual, atol=1e-12, rtol=1e-12)
