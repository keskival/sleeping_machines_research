import torch
from sleeping_machines.capacity_token_readout import frequency_readout
from sleeping_machines.frequency_token_readout import FrequencyTokenReadout


def test_default_capacity_preserves_every_parameter_and_rng():
    counts = torch.arange(1, 42); order = torch.arange(40, -1, -1)
    torch.manual_seed(6); old = FrequencyTokenReadout(16, counts, (7, 17, 30), order)
    old_rng = torch.get_rng_state()
    torch.manual_seed(6); new = frequency_readout(16, counts, (7, 17, 30), order)
    for name, value in old.state_dict().items(): assert torch.equal(value, new.state_dict()[name]), name
    assert torch.equal(old_rng, torch.get_rng_state())
    assert new.tail_widths == [8, 4, 2]


def test_full_width_tails_preserve_normalization_prior_and_target_likelihood():
    counts = torch.arange(1, 42); order = torch.arange(40, -1, -1)
    readout = frequency_readout(16, counts, (7, 17, 30), order, 16)
    assert readout.tail_widths == [16, 16, 16]
    features = torch.randn(4, 16)
    probabilities = readout.log_prob(features).exp()
    expected = (counts.float() + .1) / (counts.float() + .1).sum()
    assert torch.allclose(probabilities, expected.expand_as(probabilities), atol=1e-7, rtol=1e-6)
    assert torch.allclose(probabilities.sum(-1), torch.ones(4), atol=1e-6)
    targets = torch.tensor([40, 30, 20, 0])
    assert torch.allclose(readout.nll(features, targets), -readout.log_prob(features)[torch.arange(4), targets])


def test_wider_tail_can_use_more_independent_contextual_directions():
    counts = torch.ones(41, dtype=torch.long); order = torch.arange(41)
    for width, expected_rank in ((0, 2), (16, 11)):
        readout = frequency_readout(16, counts, (7, 17, 30), order, width)
        projection, output = readout.output.tail[-1]
        with torch.no_grad(): output.weight.normal_()
        matrix = output.weight @ projection.weight
        assert int(torch.linalg.matrix_rank(matrix)) == expected_rank
        features = torch.randn(3, 16, requires_grad=True)
        readout.nll(features, torch.tensor([30, 35, 40])).mean().backward()
        assert bool((features.grad.abs().sum(0) > 0).all())
