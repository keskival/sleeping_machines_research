import torch
from test_token_episodes import model
from sleeping_machines.event_credit_sites import sample_site, utility_weight
from sleeping_machines.paired_route_credit import paired_route_credit
from sleeping_machines.sparse_counterfactual_episodes import token_features


def test_site_rng_restores_and_later_interventions_preserve_the_prefix():
    g = torch.Generator().manual_seed(12)
    sample_site(4, 2, 2, g); saved = g.get_state()
    expected = sample_site(4, 2, 2, g)
    restored = torch.Generator(); restored.set_state(saved)
    assert sample_site(4, 2, 2, restored) == expected
    m = model(); ids = torch.tensor([[0, 1, 2, 3], [3, 4, 5, 6]])
    a, sa, _ = token_features(m, ids, generator=torch.Generator().manual_seed(9), route_credit='none')
    alt = (sa['race_winners'][2, 0, :, 0] + 1) % m.pool
    b, sb, _ = token_features(m, ids, generator=torch.Generator().manual_seed(9),
                              route_credit='none', force_site=(2, 0, 0, alt))
    assert torch.equal(a[:, :2], b[:, :2])
    assert not torch.allclose(a[:, 2:], b[:, 2:], atol=1e-12, rtol=1e-12)
    assert torch.equal(sb['race_winners'][2, 0, :, 0], alt)


def test_site_and_causal_suffix_expectations_equal_all_site_utility_gradient_sum():
    torch.manual_seed(6)
    n, depth, heads, lanes, pool = 3, 2, 2, 2, 3
    total = n * depth * heads
    logits = torch.randn(total, lanes, pool, dtype=torch.double, requires_grad=True)
    pi = logits.softmax(-1)
    costs = torch.randn(total, lanes, pool, n, dtype=torch.double)
    for site in range(total): costs[site, :, :, :site // (depth * heads)] = 0.
    exact_objective = (pi * costs.mean(-1)).sum(-1).mean(-1).sum()
    expected = torch.autograd.grad(exact_objective, logits, retain_graph=True)[0]
    surrogate = 0.
    winner = torch.zeros(lanes, dtype=torch.long)
    for site in range(total):
        token = site // (depth * heads)
        q = pi[site].detach().clone(); q[:, 0] = 0.; q /= q.sum(-1, keepdim=True)
        # Enumerate one uniformly sampled causal scoring position and every
        # eligible alternative, weighting both by their actual sampling laws.
        for position in range(token, n):
            for receiver in (1, 2):
                alt = torch.full((lanes,), receiver, dtype=torch.long)
                delta = costs[site, :, receiver, position] - costs[site, :, 0, position]
                probability = q[:, receiver]
                term = paired_route_credit(pi[site], alt, probability, delta * probability)
                surrogate += utility_weight(n, token, total) * term / total / (n - token)
    actual = torch.autograd.grad(surrogate, logits)[0]
    assert torch.allclose(expected, actual, atol=1e-12, rtol=1e-12)
