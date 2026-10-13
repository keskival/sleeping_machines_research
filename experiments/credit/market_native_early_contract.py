"""Endpoint and physical-skip contracts for B10 pre-window requests."""
import argparse, copy, json
import market_native_early_request as early
from market_streaming_clock_updates import stream as reference


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True); args = ap.parse_args()
    early.torch.set_num_threads(1); early.torch.set_default_dtype(early.torch.float64)
    model, inputs = early.restored(); t, m, path = early.load('2026-08-24', 128+3*8)
    opt = early.torch.optim.Adam(model.parameters(), lr=.0003)
    early.torch.manual_seed(19101); critic = early.NativeMarketCredit(264).double()
    critic.core.head = early.torch.nn.Linear(critic.core.total_payload, 1).double()
    early.torch.nn.init.zeros_(critic.core.head.weight)
    cop = early.torch.optim.Adam(critic.parameters(), lr=.001)
    state = early.warm(model, t, m, 128); errors = {}; saved_pair = early.pair_window
    for endpoint, bias in [('own', -1.), ('projected', 1.)]:
        critic.core.head.bias.data.fill_(bias)
        if endpoint == 'own':
            def forbidden(*args, **kwargs):
                raise AssertionError('Rejected unaudited window computed remote sensitivities')
            early.pair_window = forbidden
        else: early.pair_window = saved_pair
        native, row = early.stream((model, opt, state, critic, cop, .001), t, m, 127, 3, 8, 'frozen')
        other, ref = reference((model, opt, state), t, m, 127, 3, 8, endpoint)
        score_error = max(abs(a-b) for a, b in zip(row['block_nll'], ref['block_nll']))
        parameter_error = max(float((a-b).abs().max()) for a, b in zip(native[0].parameters(), other[0].parameters()))
        state_error = max(float((a-b).abs().max()) for a, b in zip(early.control.tensors(native[2]), early.control.tensors(other[2])))
        assert max(score_error, parameter_error, state_error) < 1e-10
        assert row['remote_windows'] == (0 if endpoint == 'own' else 3)
        errors[endpoint] = dict(score=score_error, parameters=parameter_error, state=state_error,
                                remote_windows=row['remote_windows'])
    out = early.ROOT/f'experiments/results/market/{args.tag}.json'; assert not out.exists()
    result = dict(status='completed', battle='B10', endpoint_errors=errors,
                  source_sha256={p: early.sha(early.ROOT/p) for p in [
                      'experiments/credit/market_native_early_contract.py',
                      'experiments/credit/market_native_early_request.py']},
                  scope='Three actual8target Adam updates per endpoint on TRAIN. Own endpoint raises if paired remote routine is called. Native request features accept only actor state and previously observed receipt; no target sequence argument.')
    with out.open('x') as f: json.dump(result, f, indent=2); f.write('\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
