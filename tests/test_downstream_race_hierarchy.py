import torch
from sleeping_machines.downstream_race_hierarchy import DownstreamRaceHierarchy


def model():
    torch.manual_seed(23)
    return DownstreamRaceHierarchy(classes=3, bins=2, widths=(4,4,4,4,4,4,4,4),
                                   blocks_per_stage=4, groups=2, candidates=2)


def test_replay_reproduces_all_32_races_and_restores_override():
    m = model()
    x = torch.rand(1,2,2,4,4)
    a, b = m.replay(x, 5), m.replay(x, 5)
    assert len(a['audits']) == 32
    assert torch.equal(a['logits'], b['logits'])
    for aa, bb in zip(a['audits'], b['audits']):
        assert torch.equal(aa['winners'], bb['winners'])
    m.replay(x, 5, (0,0,0,0,1))
    assert torch.equal(m.replay(x,5)['logits'], a['logits'])


def test_force_factual_choice_preserves_entire_computation():
    m = model()
    x = torch.rand(1,2,2,4,4)
    a = m.replay(x, 6)
    candidate = int(a['audits'][0]['winners'][0,0,0,0])
    b = m.replay(x, 6, (0,0,0,0,candidate))
    assert torch.equal(a['logits'], b['logits'])


def test_final_loss_credit_reaches_early_keys_without_auxiliary_heads():
    m = model().double()
    x = torch.rand(1,2,2,4,4, dtype=torch.double)
    loss, info = m.downstream_objective(x, torch.tensor([1]), 7, 0)
    loss.backward()
    grad = m.stages[0][0].keys.weight.grad
    assert torch.isfinite(grad).all() and grad.abs().sum() > 0
    assert info['audited_block'] == 0
    assert m.heads[0].weight.grad is None
    assert m.heads[-1].weight.grad.abs().sum() > 0


def test_site_importance_correction_matches_exact_gate_derivative():
    # Exact finite algebraic contract independent of neural approximation.
    scores = torch.tensor([[.2,-.4],[.7,.1]], dtype=torch.double, requires_grad=True)
    losses = torch.tensor([[2.,-1.],[.5,3.]], dtype=torch.double)
    pi = scores.softmax(-1)
    exact = torch.autograd.grad((pi*losses).sum(), scores, retain_graph=True)[0]
    sampled = []
    for site in range(2):
        surrogate = 2*((pi[site]-pi[site].detach())*losses[site]).sum()
        sampled.append(torch.autograd.grad(surrogate, scores, retain_graph=True)[0])
    assert torch.allclose(torch.stack(sampled).mean(0), exact)


def test_relational_transition_preserves_odd_shapes_and_has_mixed_child_effect():
    from sleeping_machines.downstream_race_hierarchy import RelationalComposition
    torch.manual_seed(9)
    layer = RelationalComposition(2, 4).double()
    assert layer(torch.zeros(1,2,5,7, dtype=torch.double)).shape == (1,4,3,4)
    base = torch.zeros(1,2,2,2, dtype=torch.double)
    a, b = base.clone(), base.clone()
    a[:,:,0,0] = 1
    b[:,:,1,1] = 1
    # Linear/independent child pooling has identically zero mixed difference.
    mixed = layer(a+b)-layer(a)-layer(b)+layer(base)
    assert mixed.abs().max() > 1e-8
