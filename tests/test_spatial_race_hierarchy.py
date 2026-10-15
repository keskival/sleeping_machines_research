"""G-HIER development contracts: trainable spatial composition, not timing depth."""
import torch
from sleeping_machines.spatial_race_hierarchy import SpatialRaceHierarchy, FeatureRaceBlock


def small():
    torch.manual_seed(17)
    return SpatialRaceHierarchy(classes=3, bins=2, blocks_per_stage=4, widths=(4, 6, 8, 10, 12, 14, 16, 18)).double()


def test_32_blocks_and_fixed_evidence():
    m = small().eval(); x = torch.rand(2, 2, 2, 16, 16, dtype=torch.double)
    o = m(x); assert len(o['stage_logits']) == 8
    assert sum(len(s) for s in m.stages) == 32
    assert m.work_descriptor()['representation_blocks'] == 32
    assert o['logits'].shape == (2, 3)


def test_global_gradients_reach_all_32_compositions():
    m = small().eval(); x = torch.rand(2, 2, 2, 16, 16, dtype=torch.double)
    o = m(x); torch.nn.functional.cross_entropy(o['logits'], torch.tensor([0, 2])).backward()
    for stage in m.stages:
        for block in stage:
            assert block.compose.weight.grad.abs().sum() > 0
            assert block.values.weight.grad.abs().sum() > 0


def test_spatial_arrangement_is_not_a_histogram():
    m = small().eval(); x = torch.zeros(1, 2, 2, 16, 16, dtype=torch.double)
    x[:, :, :, 2:5, 2:5] = 2
    y = x.transpose(-1, -2).flip(-1)
    assert torch.equal(x.sum((-1, -2)), y.sum((-1, -2)))
    assert not torch.allclose(m(x)['logits'], m(y)['logits'], atol=1e-10, rtol=0)


def test_exact_local_counterfactual_matches_forced_replacement():
    torch.manual_seed(9)
    b = FeatureRaceBlock(4, 2, 3).double().eval()
    x = torch.randn(1, 4, 3, 3, dtype=torch.double); out, a = b(x)
    g, k, yy, xx = 1, 0, 1, 2
    selected = a['chosen'].clone(); selected[:, g, :, yy, xx] = a['values'][:, g, k, :, yy, xx]
    forced = x + b.scale*b.mix(selected.reshape(1, 4, 3, 3))
    delta = a['values'][:, g, k, :, yy, xx]-a['chosen'][:, g, :, yy, xx]
    expected = out.mean((-1, -2)) + b.scale*torch.nn.functional.linear(delta, b.mix.weight[:, 2:, 0, 0])/9
    torch.testing.assert_close(expected, forced.mean((-1, -2)), atol=1e-12, rtol=1e-12)


def test_losing_routes_and_key_maps_receive_local_credit():
    m = small().eval(); x = torch.rand(2, 2, 2, 16, 16, dtype=torch.double)
    o = m(x); loss, _ = m.objective(o, torch.tensor([0, 2])); loss.backward()
    for stage in m.stages:
        for block in stage:
            assert block.keys.weight.grad.abs().sum() > 0
            assert block.values.weight.grad.abs().sum() > 0


def test_replay_owns_randomness_and_inputs_do_not_mutate():
    m = small().train(); x = torch.rand(1, 2, 2, 8, 8, dtype=torch.double); old = x.clone()
    a = m(x, generator=torch.Generator().manual_seed(4))
    b = m(x, generator=torch.Generator().manual_seed(4))
    torch.testing.assert_close(a['logits'], b['logits'], atol=0, rtol=0)
    assert torch.equal(x, old)


def test_first_block_has_local_spatial_dependency():
    torch.manual_seed(1)
    block = FeatureRaceBlock(4).double().eval()
    x = torch.randn(1, 4, 16, 16, dtype=torch.double)
    y = x.clone(); y[:, :, 12:, 12:] += 100
    a, _ = block(x); b, _ = block(y)
    torch.testing.assert_close(a[:, :, :5, :5], b[:, :, :5, :5], atol=0, rtol=0)


def test_local_race_key_credit_agrees_with_finite_difference():
    m=small().eval(); x=torch.rand(2,2,2,8,8,dtype=torch.double);labels=torch.tensor([0,1])
    def objective():
        out=m(x);return m.objective(out,labels,auxiliary=0,counterfactual=1)[0]
    loss=objective();loss.backward();parameter=m.stages[-1][-1].keys.bias
    analytic=float(parameter.grad[0]); original=float(parameter.detach()[0]);eps=1e-5
    with torch.no_grad():parameter[0]=original+eps
    plus=float(objective().detach())
    with torch.no_grad():parameter[0]=original-eps
    minus=float(objective().detach())
    with torch.no_grad():parameter[0]=original
    assert abs(analytic-(plus-minus)/(2*eps))<1e-7
