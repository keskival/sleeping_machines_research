import itertools
import torch
from sleeping_machines.sampled_suffix_utility import sample_positions, suffix_difference
from sleeping_machines.paired_route_credit import paired_route_credit
from sleeping_machines.token_readout import TokenReadout


def test_full_score_control_matches_original_loss_and_preserves_position_rng():
    torch.manual_seed(6)
    readout = TokenReadout(4, 7).double()
    features = torch.randn(2, 4, 4, dtype=torch.double)
    targets = torch.tensor([[0, 1, 2, 3], [3, 4, 5, 6]])
    factual = torch.randn(8, dtype=torch.double, requires_grad=True)
    g = torch.Generator().manual_seed(10); before = g.get_state()
    for count in (0, 4, 10):
        positions = sample_positions(4, count, g)
        actual = suffix_difference(readout, features, targets, factual, positions)
        expected = readout.nll(features, targets).reshape(2, 4).mean(-1) - factual.detach().reshape(2, 4).mean(-1)
        assert torch.equal(actual, expected)
        assert not actual.requires_grad
        assert torch.equal(before, g.get_state())


def test_position_sampling_expectation_preserves_actual_route_credit():
    torch.manual_seed(7)
    readout = TokenReadout(4, 7).double()
    features = torch.randn(2, 4, 4, dtype=torch.double)
    targets = torch.tensor([[0, 1, 2, 3], [3, 4, 5, 6]])
    factual = torch.rand(8, dtype=torch.double)
    logits = torch.randn(2, 3, dtype=torch.double, requires_grad=True)
    pi = logits.softmax(-1)
    alt = torch.tensor([1, 2]); q = torch.tensor([.3, .6], dtype=torch.double)
    full = suffix_difference(readout, features, targets, factual, torch.arange(4))
    expected = torch.autograd.grad(paired_route_credit(pi, alt, q, full), logits, retain_graph=True)[0]
    for count in (1, 2, 3):
        subsets = list(itertools.combinations(range(4), count))
        average = sum(paired_route_credit(pi, alt, q, suffix_difference(
            readout, features, targets, factual, torch.tensor(s))) for s in subsets) / len(subsets)
        actual = torch.autograd.grad(average, logits, retain_graph=True)[0]
        assert torch.allclose(expected, actual, atol=1e-12, rtol=1e-12)


def test_position_rng_resumes_exactly_and_has_no_duplicate_positions():
    g = torch.Generator().manual_seed(9)
    first = sample_positions(16, 4, g); saved = g.get_state()
    second = sample_positions(16, 4, g)
    restored = torch.Generator(); restored.set_state(saved)
    assert torch.equal(second, sample_positions(16, 4, restored))
    assert first.unique().numel() == 4
    assert bool((first[1:] > first[:-1]).all())
