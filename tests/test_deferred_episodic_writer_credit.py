import copy
import io
import torch
from unittest.mock import patch
from tests.test_native_indexed_cause_head import model_and_bank
from tests.test_native_query_migration import identical
from experiments.native_query_migration import migrate
from experiments.r1_local_native_deep import LocalNativeDeepLM
from sleeping_machines.native_indexed_cause_head import indexed_cause_nll
from sleeping_machines.episodic_writer_cause_head import indexed_cause_nll_with_writer_features
from sleeping_machines.producer_replay_native_race import enable_producer_credit
from sleeping_machines.native_cause_bank import NativeCauseBank
from sleeping_machines.deferred_episodic_writer_credit import enable_deferred_episodic_credit, deferred_writer_law, EpisodicCreditBatch


def test_explicit_feature_channel_preserves_law_and_head_derivatives():
    model, bank = model_and_bank()
    other = copy.deepcopy(model)
    views = torch.randn(3, 8, requires_grad=True)
    alternate_views = views.detach().clone().requires_grad_()
    candidates = bank.query(1, [('unit', 0)], 3)
    torch.manual_seed(81)
    loss, aux, _ = indexed_cause_nll(model.causes, views, 4, candidates, explore=True)
    (loss.mean()+aux).backward()
    leaves = candidates['features'].detach().clone().requires_grad_()
    torch.manual_seed(81)
    other_loss, other_aux, _ = indexed_cause_nll_with_writer_features(
        other.causes, alternate_views, 4, dict(candidates, _writer_features=leaves), explore=True)
    (other_loss.mean()+other_aux).backward()
    assert torch.equal(loss, other_loss) and torch.equal(aux, other_aux)
    torch.testing.assert_close(views.grad, alternate_views.grad, rtol=0, atol=0)
    for (n, p), (m, q) in zip(model.causes.named_parameters(), other.causes.named_parameters()):
        assert n == m and (p.grad is None) == (q.grad is None)
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
    assert leaves.grad.norm() > 0


def acquired():
    torch.manual_seed(18531)
    model = LocalNativeDeepLM(8, torch.arange(8)//2, torch.arange(8).reshape(4,2),
                              width=8, depth=3, heads=2, pool=3, concept_units=16,
                              topk=2, explore=1, dk=4, packed=True, bank_capacity=16)
    opt = torch.optim.AdamW(model.parameters(), lr=.001)
    state = model.new_state()
    state['cause_bank'] = NativeCauseBank(3, 8, capacity=16)
    state['cause_inbox'] = torch.zeros(3, 8)
    model, opt, state, _ = migrate(model, opt, state, publication_capacity=2)
    enable_producer_credit(model, state)
    return model, opt, state


def test_deferred_writer_credit_preserves_forecasts_and_reaches_producers():
    baseline, _, state = acquired()
    model = copy.deepcopy(baseline); other = copy.deepcopy(state)
    enable_deferred_episodic_credit(model, other)
    tokens = torch.tensor([1,2,1,2,1,3,1,2,1])
    g = torch.Generator().manual_seed(11); a = torch.Generator().manual_seed(12)
    bg = torch.Generator().manual_seed(11); ba = torch.Generator().manual_seed(12)
    torch.manual_seed(80)
    before, _, _, _, _ = baseline.stream_nll(tokens[:-1], tokens[1:], state, bg, ba)
    torch.manual_seed(80)
    actual, objective, _, packets, _ = model.stream_nll(tokens[:-1], tokens[1:], other, g, a)
    assert torch.equal(before, actual)
    assert torch.equal(g.get_state(), bg.get_state()) and torch.equal(a.get_state(), ba.get_state())
    objective.backward(); model.deliver_credit(packets, other)
    assert other['episodic_writer_replays'] > 0
    assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
    assert model.core.mix.grad is not None and model.core.mix.grad.norm() > 0
    assert other['cause_bank'].count == state['cause_bank'].count
    assert torch.equal(other['cause_bank'].features, state['cause_bank'].features)
    assert not hasattr(other['cause_bank'], '_writer_context')


def test_deferred_writer_complete_update_recovery_and_unchanged_evaluation():
    model, opt, state = acquired()
    enable_deferred_episodic_credit(model, state)
    tokens = torch.tensor([1,2,1,2,3,1,2,1,3])
    g = torch.Generator().manual_seed(13); a = torch.Generator().manual_seed(14)
    def update(start, end):
        opt.zero_grad(set_to_none=True)
        nll, objective, _, packets, _ = model.stream_nll(tokens[start:end], tokens[start+1:end+1], state, g, a)
        objective.backward(); model.deliver_credit(packets, state); opt.step()
        return nll.detach().clone()
    update(0, 2)
    buffer = io.BytesIO()
    torch.save((model.state_dict(), opt.state_dict(), state, g.get_state(), a.get_state(), torch.get_rng_state()), buffer)
    expected_loss = update(2, 5)
    expected = copy.deepcopy((model.state_dict(), opt.state_dict(), state, g.get_state(), a.get_state(), torch.get_rng_state()))
    buffer.seek(0)
    weights, moments, state, gs, als, rng = torch.load(buffer, weights_only=False)
    model.load_state_dict(weights); opt.load_state_dict(moments); g.set_state(gs); a.set_state(als); torch.set_rng_state(rng)
    assert torch.equal(expected_loss, update(2, 5))
    assert identical(expected, (model.state_dict(), opt.state_dict(), state, g.get_state(), a.get_state(), torch.get_rng_state()))
    replays = state['episodic_writer_replays']
    with torch.no_grad():
        model.stream_nll(tokens[5:7], tokens[6:8], state, g, a, learn=False)
    assert state['episodic_writer_replays'] == replays


def test_deferred_sampling_mean_matches_all_available_producer_pullbacks():
    model, _, state = acquired()
    g = torch.Generator().manual_seed(31); a = torch.Generator().manual_seed(32)
    with torch.no_grad():
        model.stream_nll(torch.tensor([1,2,1]), torch.tensor([2,1,2]), state, g, a, learn=False)
    candidates = state['cause_bank'].query(1, [], 2)
    views = torch.randn(3, 8)
    eligible = [(0, d, state['publications'][d][-1]) for d in [1, 2]]
    features = candidates['features'].detach().requires_grad_()
    loss, _, _ = indexed_cause_nll_with_writer_features(model.causes, views, 2,
                                                       dict(candidates, _writer_features=features))
    cotangent = torch.autograd.grad(loss.mean(), features)[0]
    expected = torch.zeros_like(model.core.mix)
    for i, d, key in eligible:
        value = model.core.read_publication(state, key, state['bindings'].entries[key].available, [], learn=True)
        expected += torch.autograd.grad(value, model.core.mix, grad_outputs=cotangent[i,d])[0]
    norms = torch.stack([cotangent[i,d].double().norm() for i,d,_ in eligible])
    probabilities = .9*norms/norms.sum()+.1/len(eligible)
    state.update(episodic_writer_rng=torch.Generator().manual_seed(33).get_state(),
                 episodic_writer_replays=0, episodic_writer_zero_cotangent_events=0,
                 episodic_writer_max_inverse_probability=0.)
    actual = torch.zeros_like(expected)
    for selected in range(len(eligible)):
        model.zero_grad(set_to_none=True)
        snapshot = {k:v for k,v in state.items() if isinstance(v,int)}
        snapshot['bindings'] = copy.copy(state['bindings'])
        snapshot['bindings'].entries = dict(state['bindings'].entries)
        snapshot['producer_records'] = dict(state['producer_records'])
        context = dict(core=model.core, state=state, batch=EpisodicCreditBatch(), floor=.1,
                       versions=[(p,p._version) for p in model.parameters()])
        candidate = dict(candidates, _writer_credit=(context,snapshot,eligible))
        with patch('torch.multinomial', return_value=torch.tensor([selected])):
            loss, _, _ = deferred_writer_law(model.causes, views, 2, candidate)
            loss.mean().backward()
        actual += probabilities[selected].float()*model.core.mix.grad
    torch.testing.assert_close(actual, expected, rtol=4e-5, atol=3e-6)
