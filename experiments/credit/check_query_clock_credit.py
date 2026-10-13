"""B10 integrated query-credit null, activity and actual-learner smoke checks."""
import argparse
import copy
import json
import statistics
import time
import market_addressed_clock_credit_acquisition as acquisition
from native_query_clock_credit import QueryClockCredit
from native_addressed_clock_credit import guided, support

ROOT, torch, np, engine = acquisition.ROOT, acquisition.torch, acquisition.np, acquisition.engine


def completed_cohort_diagnosis(inputs):
    """Descriptive quarter/teacher diagnostics; no causal or significance test."""
    rows = []
    for seed in (19124, 19129, 19130):
        tag = ('curie_b10_addressed_clock_20261013_v2_acquisition' if seed == 19124 else
               f'curie_b10_addressed_clock_20261013_confirm_s{seed}_v1_addressed')
        path = ROOT / f'experiments/results/market/{tag}.json'
        packet = json.loads(path.read_text())
        assert packet['status'] == 'completed' and packet['dev']['targets'] == 16384
        for name, digest in packet['source_sha256'].items():
            assert engine.acquired.sha(ROOT / name) == digest
        inputs[str(path.relative_to(ROOT))] = engine.acquired.sha(path)
        source = packet['acquisition']['rows']
        gains = []
        for quarter in range(4):
            start, stop = quarter * 256, (quarter + 1) * 256
            gains.append(statistics.mean(.5 * (ref['producer_nll'] + ref['future_nll'] -
                                                dev['producer_nll'] - dev['future_nll'])
                          for dev, ref in zip(packet['dev']['rows'][start:stop],
                                              packet['reference']['rows'][start:stop])))
        assert abs(statistics.mean(gains) - packet['dev_gain']) < 1e-12
        rows.append(dict(parent_seed=seed, dev_gain=packet['dev_gain'], quarter_gains=gains,
                         critic_gradient_norm_median=statistics.median(r['critic_grad_norm'] for r in source),
                         critic_gradient_norm_max=max(r['critic_grad_norm'] for r in source),
                         antithetic_abs_contrast_median=statistics.median(abs(r['antithetic_contrast']) for r in source),
                         antithetic_abs_contrast_max=max(abs(r['antithetic_contrast']) for r in source)))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(19131)
    begin = time.monotonic()
    acquisition.teacher.guided = guided
    model, inputs = engine.acquired.restored()
    diagnosis = completed_cohort_diagnosis(inputs)
    path = ROOT / 'experiments/results/market/curie_b10_future_updates_20261013_v1_fit_critic.pt'
    packet = torch.load(path, map_location='cpu', weights_only=False)
    inputs[str(path.relative_to(ROOT))] = engine.acquired.sha(path)
    parent = acquisition.native.NativeMarketCredit(264).double()
    parent.core.head = torch.nn.Linear(parent.core.total_payload, 4).double()
    parent.load_state_dict(packet['state_dict'])
    parent_scale = np.array(packet['scale'])
    scale = np.repeat(parent_scale, 16) / 4
    critic = QueryClockCredit(parent, parent_scale)
    assert torch.all(torch.stack([support(i) for i in range(32)]).sum(0) == 2)

    # Exact parent forecasts and global tape despite fewer private native events.
    t, m, path = engine.acquired.load('2026-08-25', 128 + 48)
    inputs[str(path.relative_to(ROOT))] = engine.acquired.sha(path)
    torch.manual_seed(19125)
    _, actual = acquisition.native.stream(model, critic, parent, scale, t, m, 3, False)
    endpoint = torch.get_rng_state().clone()
    torch.manual_seed(19125)
    reference = engine.stream(model, parent, parent_scale, t, m, 3, 'learned_refresh')
    assert torch.equal(torch.get_rng_state(), endpoint)
    error = max(abs(a[key] - b[key]) for a, b in zip(actual['rows'], reference['rows'])
                for key in ('producer_nll', 'future_nll'))
    assert error == 0

    # Measure native event counts on a real issued optimizer context.
    state = engine.acquired.warm(model, t, m, 128)
    old = engine.acquired.control.detach_state(state)
    end, grads, scores = engine.window(model, state, t, m, 127, 8, 'own')
    opt = torch.optim.Adam(model.parameters(), lr=.0003)
    features, events, operator = acquisition.token(model, opt, old, end, grads, scores, t, m, 127)
    before = torch.get_rng_state().clone()
    with torch.no_grad():
        output, carried = critic(features, events, None, operator)
    after = torch.get_rng_state().clone()
    assert critic.receipt.events == 4 and critic.parent.events == 2
    assert carried.receipt.state.selected_updates + carried.parent.state.selected_updates == 24
    assert carried.receipt.state.candidate_scores + carried.parent.state.candidate_scores == 48
    torch.set_rng_state(before)
    with torch.no_grad():
        parent_output, _ = parent(features, events, None)
    assert torch.equal(torch.get_rng_state(), after)
    assert torch.equal(output[0], parent_output[0].repeat_interleave(16))

    # Three source receipts verify actual Adam and coupled finite-future worlds.
    torch.manual_seed(19131)
    t, m, path = engine.acquired.load('2026-08-24', 128 + 96)
    inputs[str(path.relative_to(ROOT))] = engine.acquired.sha(path)
    fitted, source = acquisition.fit(copy.deepcopy(model), QueryClockCredit(parent, parent_scale),
                                     scale, t, m, 3, True)
    assert all(row['paired_zero_action_parity'] == 0 and row['first_future_parity'] < 1e-9
               for row in source['rows'])
    names = ['experiments/credit/check_query_clock_credit.py',
             'experiments/credit/native_query_clock_credit.py',
             'experiments/credit/native_addressed_clock_credit.py',
             'experiments/credit/native_optimizer_receipts.py',
             'experiments/credit/market_addressed_clock_credit_acquisition.py',
             'experiments/credit/native_receipt_evaluation.py',
             'experiments/credit/native_market_credit.py',
             'experiments/credit/market_actual_update_teacher.py']
    result = dict(status='completed', battle='B10', null_forecast_error=error,
                  null_factual_rng_parity=True, source_smoke=source,
                  completed_addressed_cohort_diagnosis=diagnosis,
                  native_events_per_decision=6, candidate_scores_per_decision=48,
                  selected_writes_per_decision=24,
                  total_parameters=sum(p.numel() for p in fitted.parameters()),
                  trainable_parameters=sum(p.numel() for p in fitted.parameters() if p.requires_grad),
                  input_sha256=inputs,
                  source_sha256={name: engine.acquired.sha(ROOT / name) for name in names},
                  wall_s=time.monotonic() - begin,
                  scope='48 DEV targets for exact zero-head parity and three TRAIN receipts for numerical smoke. '
                        'No fitted quality comparison, no TEST, no whole-work/physical-traffic/energy claim. '
                        'All-mode receipt preprocessing, parent and actor/optimizer work remain paid.')
    with (ROOT / f'experiments/results/market/{args.tag}.json').open('x') as handle:
        json.dump(result, handle, indent=2)
        handle.write('\n')
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ('source_smoke', 'input_sha256', 'source_sha256')}), flush=True)


if __name__ == '__main__':
    main()
