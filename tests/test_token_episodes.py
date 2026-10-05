import torch
from sleeping_machines.addressed_event_heads import AddressedEventHeads
from sleeping_machines.fast_native_core import fast_class
from sleeping_machines.carried_episodes import carried_logits
from sleeping_machines.compiled_episodes import layer_step
from sleeping_machines.token_episodes import token_features, detach
from sleeping_machines.token_readout import TokenReadout


def model():
    torch.manual_seed(3)
    return fast_class(AddressedEventHeads)(sources=1, content_dim=7, classes=7, payload=4,
                                         depth=2, heads=2, pool=3).double()


def test_lookup_matches_one_hot_logits_and_every_parameter_gradient():
    m = model(); ids = torch.tensor([[0, 1, 2, 4], [3, 1, 6, 0]])
    marks = torch.nn.functional.one_hot(ids, 7).double()
    old, _ = carried_logits(m, torch.arange(4).double().expand(2, 4), marks,
                            seed=19, step=layer_step, route_credit='linear')
    old.square().sum().backward()
    grads = {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}
    m.zero_grad(set_to_none=True)
    features, _, _ = token_features(m, ids, generator=torch.Generator().manual_seed(19))
    new = m.head(features)
    assert torch.allclose(new, old, atol=1e-12, rtol=1e-12)
    new.square().sum().backward()
    for name, p in m.named_parameters():
        if name in grads:
            assert torch.allclose(p.grad, grads[name], atol=1e-10, rtol=1e-10), name


def test_stochastic_chunk_partition_preserves_features_state_and_gradients():
    m = model(); ids = torch.tensor([[0, 1, 2, 3, 4], [3, 1, 6, 0, 2]])
    whole, ws, _ = token_features(m, ids, generator=torch.Generator().manual_seed(11))
    gen = torch.Generator().manual_seed(11)
    a, state, _ = token_features(m, ids[:, :2], generator=gen)
    b, state, _ = token_features(m, ids[:, 2:], state, generator=gen)
    assert torch.equal(whole, torch.cat([a, b], 1))
    assert all(torch.equal(x, y) for x, y in zip(ws['mem'], state['mem']))
    g1 = torch.autograd.grad(whole.square().sum(), tuple(m.parameters()), allow_unused=True)
    g2 = torch.autograd.grad(torch.cat([a, b], 1).square().sum(), tuple(m.parameters()), allow_unused=True)
    for x, y in zip(g1, g2):
        if x is not None:
            assert torch.allclose(x, y, atol=1e-10, rtol=1e-10)
    ds = detach(state)
    assert all(not x.requires_grad and torch.equal(x, y) for x, y in zip(ds['mem'], state['mem']))


def test_eos_resets_only_its_lane_and_erases_prior_content():
    m = model()
    a, sa, _ = token_features(m, torch.tensor([[1, 2, 6, 3], [0, 1, 2, 3]]), deterministic=True, eos=6)
    b, sb, _ = token_features(m, torch.tensor([[4, 0, 6, 3], [0, 1, 2, 3]]), deterministic=True, eos=6)
    assert torch.equal(a[0, 2:], b[0, 2:]) and torch.equal(a[1], b[1])
    assert torch.equal(sa['position'], torch.tensor([2., 4.], dtype=torch.float64))


def test_adaptive_readout_is_normalized_and_target_likelihood_matches_full_distribution():
    torch.manual_seed(1)
    for cutoffs in ((), (3, 6)):
        h = TokenReadout(8, 9, cutoffs, torch.randperm(9)).double()
        x = torch.randn(11, 8, dtype=torch.float64, requires_grad=True); y = torch.arange(11) % 9
        lp = h.log_prob(x)
        assert torch.allclose(lp.exp().sum(-1), torch.ones(11).double(), atol=1e-12)
        assert torch.allclose(h.nll(x, y), -lp[torch.arange(11), y], atol=1e-12)
        h.nll(x, y).mean().backward()
        assert torch.isfinite(x.grad).all()


def test_optimizer_checkpoint_restores_numeric_state_generator_and_next_update(tmp_path):
    m = model(); opt = torch.optim.AdamW(m.parameters(), lr=.002)
    gen = torch.Generator().manual_seed(52)
    ids = torch.tensor([[0, 1, 2], [3, 4, 5]])
    x, st, _ = token_features(m, ids, generator=gen)
    m.head(x).square().mean().backward(); opt.step(); opt.zero_grad(set_to_none=True)
    st = detach(st)
    path = tmp_path/'checkpoint.pt'
    torch.save(dict(model=m.state_dict(), optimizer=opt.state_dict(),state=st,generator=gen.get_state()),path)
    def update(mm, oo, ss, gg):
        xx, ss, _ = token_features(mm, ids, ss, generator=gg)
        zz = mm.head(xx); zz.square().mean().backward(); oo.step()
        return zz.detach(), ss
    ref, rs = update(m,opt,st,gen)
    ck = torch.load(path,weights_only=False); restored = model()
    restored.load_state_dict(ck['model']); ro = torch.optim.AdamW(restored.parameters(),lr=.002)
    ro.load_state_dict(ck['optimizer']); rg = torch.Generator(); rg.set_state(ck['generator'])
    actual, ss = update(restored,ro,ck['state'],rg)
    assert torch.equal(ref,actual)
    assert all(torch.equal(x,y) for x,y in zip(m.parameters(),restored.parameters()))
    assert all(torch.equal(x,y) for x,y in zip(rs['mem'],ss['mem']))
    assert torch.equal(gen.get_state(),rg.get_state())


def test_frequency_readout_starts_at_train_only_prior_and_remains_normalized():
    from sleeping_machines.frequency_token_readout import FrequencyTokenReadout
    counts = torch.tensor([20, 0, 10, 4, 1, 30, 2, 0, 1])
    order = counts.argsort(descending=True,stable=True)
    for cuts in ((), (3, 6)):
        h = FrequencyTokenReadout(8,counts,cuts,order).double()
        x = torch.randn(4,8,dtype=torch.float64)
        expected = (counts.double()+.1)/(counts.sum()+.9)
        assert torch.allclose(h.log_prob(x).exp(), expected.expand(4,9),atol=1e-8)
        loss = h.nll(x,torch.tensor([0,1,5,8])).mean()
        loss.backward()
        opt = torch.optim.AdamW(h.parameters(),lr=.001); opt.step()
        assert torch.allclose(h.log_prob(x).exp().sum(-1),torch.ones(4).double(),atol=1e-12)
