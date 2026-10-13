"""B10: causal pre-window native admission with physically skipped remote traces."""
import argparse, copy, json, resource, time
from market_native_remote_admission_selected import (
    ROOT, torch, np, restored, load, warm, window, control, pair_window,
    NativeMarketCredit, flat, step, proposal, future_score, sha,
)


def context(model, state, previous):
    """No input from the unobserved window; missing history has an explicit flag."""
    current = flat(state)
    earlier = current if previous is None else previous['start']
    descriptor = (current-earlier).clone()
    descriptor[:8] = 0
    feedback = current.new_zeros(8)
    if previous is not None:
        descriptor[:4] = previous['own_projection'].asinh()
        descriptor[4] = 1  # previous observation exists, not a computed zero
        feedback = previous['feedback']
    features = torch.cat((current, earlier, descriptor))[None]
    return features, torch.cat((features[0], feedback))[None, None]


def remember(model, old, own, t, m, k, group, scores):
    by = dict(zip((n for n, _ in model.named_parameters()), own))
    projection = torch.stack([by[f'layers.{j}.{key}'].sum()/np.sqrt(l.n)
                              for j, l in enumerate(model.layers)
                              for key in ['log_rate', 'freq']]).detach()
    feedback = projection.new_zeros(8)
    feedback[0] = torch.log1p((t[k+group]-t[k])/model.scale)
    feedback[1:7] = torch.nn.functional.one_hot(m[k+1:k+group+1], 6).double().mean(0)
    feedback[7] = float(np.mean(scores))
    return dict(start=flat(old), own_projection=projection, feedback=feedback)


def stream(bundle, t, m, start, blocks, group, arm, acquisition=False):
    model, opt, state, critic, cop, scale = copy.deepcopy(bundle)
    carry = previous = pending = None
    rows, requests, remote, labels = [], [], [], []
    begin = time.monotonic()
    phase = {k: 0. for k in ['request', 'credit', 'teacher', 'proposal', 'optimizer']}
    interval = 2 if acquisition else 16
    for block in range(blocks):
        k = start+block*group
        teach = (acquisition or arm == 'online') and block % interval == 0 and block+1 < blocks
        # Every teaching frame is followed by a nonteaching frame: no pending
        # activation survives an intervening controller parameter mutation.
        assert not (teach and pending is not None)
        tick = time.monotonic()
        if acquisition or arm in ['frozen', 'online']:
            features, events = context(model, state, previous)
            critic.core.train(teach)
            with torch.set_grad_enabled(teach):
                output, carry = critic(features, events, carry)
            q = float(output.detach().item() > 0)
        else:
            q = float(arm == 'projected')
        phase['request'] += time.monotonic()-tick
        # q is fixed before reading/scoring the first outcome in this window.
        need_remote = bool(q or teach)
        tick = time.monotonic()
        old = control.detach_state(state)
        if need_remote:
            end, own, projected, scores = pair_window(model, state, t, m, k, group)
        else:
            end, own, scores = window(model, state, t, m, k, group, 'own')
            projected = None
        if acquisition:
            exact_end, grads, exact_scores = window(model, copy.deepcopy(old), t, m, k, group, 'bptt')
            assert max(abs(a-b) for a, b in zip(scores, exact_scores)) < 1e-12
        else:
            grads = projected if q else own
        phase['credit'] += time.monotonic()-tick
        if pending is not None:
            tick = time.monotonic()
            prediction, candidates, decision = pending
            delta = future_score(candidates[0], t, m, k, group)-future_score(candidates[1], t, m, k, group)
            target = float(np.clip(delta/scale, -5., 5.))
            cop.zero_grad(set_to_none=True)
            (prediction-target).square().mean().backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1., error_if_nonfinite=True)
            cop.step()
            scale = max(1e-5, .95*scale+.05*abs(delta))
            labels.append(dict(decision_block=decision, observed_block=block, delta=delta))
            pending = None
            phase['teacher'] += time.monotonic()-tick
        if teach:
            tick = time.monotonic()
            pending = (output, [proposal(model, opt, end, g) for g in [own, projected]], block)
            phase['proposal'] += time.monotonic()-tick
        previous = remember(model, old, own, t, m, k, group, scores)
        tick = time.monotonic()
        step(model, opt, grads)
        state = control.detach_state(end)
        phase['optimizer'] += time.monotonic()-tick
        rows.append(float(np.mean(scores))); requests.append(q); remote.append(need_remote)
    assert pending is None
    row = dict(arm=arm, nll=float(np.mean(rows)), targets=blocks*group,
               wall_s=time.monotonic()-begin, phase_wall_s=phase,
               block_nll=rows, requests=requests, remote_computed=remote,
               remote_windows=sum(remote), requested_windows=sum(requests),
               teachers=len(labels), teacher_labels=labels)
    return (model, opt, state, critic, cop, scale), row


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True)
    ap.add_argument('--mode', choices=['smoke', 'fit'], required=True)
    args = ap.parse_args(); torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
    out = ROOT/f'experiments/results/market/{args.tag}.json'; assert not out.exists()
    begin = time.monotonic(); model, inputs = restored()
    group, acquisition, reuse = (8, 4, 4) if args.mode == 'smoke' else (32, 64, 512)
    torch.manual_seed(19101)
    critic = NativeMarketCredit(264).double()
    critic.core.head = torch.nn.Linear(critic.core.total_payload, 1).double()
    torch.nn.init.zeros_(critic.core.head.weight); torch.nn.init.zeros_(critic.core.head.bias)
    opt = torch.optim.Adam(model.parameters(), lr=.0003)
    cop = torch.optim.Adam(critic.parameters(), lr=.001)
    t, m, path = load('2026-08-24', 128+acquisition*group)
    inputs[str(path.relative_to(ROOT))] = sha(path)
    bundle, source = stream((model, opt, warm(model, t, m, 128), critic, cop, .001),
                            t, m, 127, acquisition, group, 'online', True)
    t, m, path = load('2026-08-25', 128+reuse*group)
    inputs[str(path.relative_to(ROOT))] = sha(path)
    bundle = (*bundle[:2], warm(bundle[0], t, m, 128), *bundle[3:])
    rows = []
    for arm in ['own', 'projected', 'frozen', 'online']:
        # Identical initial native clock stream, isolated from other arms' length.
        torch.manual_seed(19111)
        _, row = stream(bundle, t, m, 127, reuse, group, arm); rows.append(row)
    files = ['experiments/credit/market_native_early_request.py',
             'experiments/market/NATIVE_EARLY_REQUEST_PROTOCOL.md',
             'experiments/credit/market_native_remote_admission_selected.py',
             'experiments/credit/native_market_credit.py',
             'experiments/credit/market_streaming_clock_updates.py',
             'experiments/market/paired_clock_credit.py',
             'experiments/market/projected_clock_state.py',
             'experiments/market/local_eligibility_state.py']
    result = dict(status='completed', battle='B10', args=vars(args), acquisition=source,
                  rows=rows, source_sha256={p: sha(ROOT/p) for p in files}, input_sha256=inputs,
                  wall_s=time.monotonic()-begin, peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  scope='Pre-window requests from prior observed context only. Deterministic full-support source audits every2windows; online deployment audits every16windows, not a propensity-corrected random risk estimator. Frozen deployment has no audits. Common2048TRAIN exact actor acquisition; reused Aug25DEV; no TEST or efficiency/benchmark claim inferred from one run. Native private banks and current-frame counterfactual learning retained.')
    with out.open('x') as f: json.dump(result, f, indent=2); f.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ['rows', 'acquisition', 'source_sha256', 'input_sha256']}), flush=True)
    for row in rows: print(json.dumps({k: v for k, v in row.items() if k not in ['block_nll', 'requests', 'remote_computed', 'teacher_labels']}), flush=True)


if __name__ == '__main__': main()
