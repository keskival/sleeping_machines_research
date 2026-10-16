import copy
import io
import torch
from sleeping_machines.native_future_update_credit import NativeFutureUpdateCredit
from tests.test_native_query_migration import identical


def test_native_future_action_noop_support_and_causal_quote():
    torch.manual_seed(19171)
    critic = NativeFutureUpdateCredit(8, depth=3, heads=2, pool=3)
    state = critic.new_state(); g = torch.Generator().manual_seed(91); a = torch.Generator().manual_seed(92)
    vectors = [torch.randn(8) for _ in range(3)]
    quote, state, packets = critic.quote(*vectors, .5, .001, state, g, a)
    assert quote.shape == (3,) and float(quote[1]) == 0
    choice, probability = critic.choose(quote, torch.Generator().manual_seed(93))
    assert 0 <= choice < 3 and .2/3 <= probability <= 1
    optimizer = torch.optim.AdamW(critic.parameters(), lr=.001)
    before = copy.deepcopy(critic.state_dict())
    record = critic.learn_revealed(quote[0], -.01, packets, state, optimizer)
    assert record['target_advantage'] < 0
    assert any(not torch.equal(before[k], v) for k, v in critic.state_dict().items())
    assert state['credit_projections'] > 0


def test_native_future_credit_complete_recovery():
    torch.manual_seed(19172)
    critic = NativeFutureUpdateCredit(8, depth=3, heads=2, pool=3)
    optimizer = torch.optim.AdamW(critic.parameters(), lr=.001)
    state = critic.new_state(); g = torch.Generator().manual_seed(94); a = torch.Generator().manual_seed(95)
    values = [torch.randn(8) for _ in range(3)]
    def update():
        quote, _, packets = critic.quote(*values, .2, .001, state, g, a)
        metrics = critic.learn_revealed(quote[2], .02, packets, state, optimizer)
        return quote.detach(), metrics
    update()
    buffer = io.BytesIO()
    torch.save((critic.state_dict(), optimizer.state_dict(), state, g.get_state(), a.get_state()), buffer)
    expected = update()
    final = copy.deepcopy((critic.state_dict(), optimizer.state_dict(), state, g.get_state(), a.get_state()))
    buffer.seek(0)
    weights, moments, state, gs, als = torch.load(buffer, weights_only=False)
    critic.load_state_dict(weights); optimizer.load_state_dict(moments); g.set_state(gs); a.set_state(als)
    actual = update()
    assert torch.equal(expected[0], actual[0]) and expected[1] == actual[1]
    assert identical(final, (critic.state_dict(), optimizer.state_dict(), state, g.get_state(), a.get_state()))
