"""B10: causal pre-window native admission with physically skipped remote traces."""
import argparse, copy, json, resource, time
from market_clock_context_contract import clock_context_window
import market_native_early_request as original
from market_complete_work_probe_v2 import LikelihoodAudit
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
    teacher_replays = 0
    teacher_replay_flops = 0
    reused_factual_windows = 0
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
        compact = acquisition and not teach
        need_remote = bool(teach if acquisition else (q or teach))
        tick = time.monotonic()
        old = control.detach_state(state)
        if compact:
            end, grads, scores, clock_projection = clock_context_window(model, state, t, m, k, group)
            own = projected = None
        elif need_remote:
            end, own, projected, scores = pair_window(model, state, t, m, k, group)
        else:
            end, own, scores = window(model, state, t, m, k, group, 'own')
            projected = None
        if acquisition and not compact:
            exact_end, grads, exact_scores = window(model, copy.deepcopy(old), t, m, k, group, 'bptt')
            assert max(abs(a-b) for a, b in zip(scores, exact_scores)) < 1e-12
        elif not acquisition:
            grads = projected if q else own
        phase['credit'] += time.monotonic()-tick
        if pending is not None:
            tick = time.monotonic()
            prediction, candidates, decision, selected = pending
            if acquisition:
                delta = future_score(candidates[0], t, m, k, group)-future_score(candidates[1], t, m, k, group)
                teacher_replays += 2
            else:
                chosen = candidates[int(selected)]
                assert all(torch.equal(a,b) for a,b in zip(model.parameters(),chosen[0].parameters()))
                assert all(torch.equal(a,b) for a,b in zip(control.tensors(old),control.tensors(chosen[1])))
                factual = float(np.mean(scores))
                with LikelihoodAudit() as audit:
                    alternative = future_score(candidates[1-int(selected)], t, m, k, group)
                work = audit.result();assert not work['unsupported_floating_operators']
                teacher_replay_flops += work['arithmetic_flops']
                delta = factual-alternative if selected==0 else alternative-factual
                teacher_replays += 1;reused_factual_windows += 1
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
            pending = (output, [proposal(model, opt, end, g) for g in [own, projected]], block, q)
            phase['proposal'] += time.monotonic()-tick
        if compact:
            feedback = flat(old).new_zeros(8)
            feedback[0] = torch.log1p((t[k+group]-t[k])/model.scale)
            feedback[1:7] = torch.nn.functional.one_hot(m[k+1:k+group+1], 6).double().mean(0)
            feedback[7] = float(np.mean(scores))
            previous = dict(start=flat(old), own_projection=torch.as_tensor(clock_projection), feedback=feedback)
        else:
            previous = remember(model, old, own, t, m, k, group, scores)
        tick = time.monotonic()
        step(model, opt, grads)
        state = control.detach_state(end)
        phase['optimizer'] += time.monotonic()-tick
        rows.append(float(np.mean(scores))); requests.append(q); remote.append(need_remote)
    assert pending is None
    row = dict(teacher_replays=teacher_replays,teacher_replay_arithmetic_flops=teacher_replay_flops,reused_factual_windows=reused_factual_windows,arm=arm, nll=float(np.mean(rows)), targets=blocks*group,
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
    initial = (model, opt, warm(model, t, m, 128), critic, cop, .001)
    source_rng = torch.get_rng_state().clone()
    bundle, source = stream(initial,
                            t, m, 127, acquisition, group, 'online', True)
    torch.set_rng_state(source_rng)
    reference, original_source = original.stream(initial, t, m, 127, acquisition, group, 'online', True)
    actor_error = max(float((a-b).abs().max()) for a,b in zip(bundle[0].parameters(), reference[0].parameters()))
    critic_error = max(float((a-b).abs().max()) for a,b in zip(bundle[3].parameters(), reference[3].parameters()))
    moment_error = max(float((bundle[1].state[a][key]-reference[1].state[b][key]).abs().max()) for a,b in zip(bundle[0].parameters(),reference[0].parameters()) for key in ['step','exp_avg','exp_avg_sq'])
    assert max(actor_error,critic_error,moment_error)<1e-12
    source_parity=dict(actor=actor_error,critic=critic_error,actor_moments=moment_error)
    t, m, path = load('2026-08-25', 128+reuse*group)
    inputs[str(path.relative_to(ROOT))] = sha(path)
    bundle = (*bundle[:2], warm(bundle[0], t, m, 128), *bundle[3:])
    rows = []
    for arm in ['online']:
        # Identical initial native clock stream, isolated from other arms' length.
        torch.manual_seed(19111)
        _, row = stream(bundle, t, m, 127, reuse, group, arm); rows.append(row)
    files = ['experiments/credit/market_native_factual_teacher_reuse.py','experiments/credit/market_complete_work_probe_v2.py','sleeping_machines/operation_audit.py','experiments/credit/market_native_compact_acquisition.py', 'experiments/credit/market_clock_context_contract.py', 'experiments/credit/market_native_early_request.py',
             'experiments/market/NATIVE_EARLY_REQUEST_PROTOCOL.md',
             'experiments/credit/market_native_remote_admission_selected.py',
             'experiments/credit/native_market_credit.py',
             'experiments/credit/market_streaming_clock_updates.py',
             'experiments/market/paired_clock_credit.py',
             'experiments/market/projected_clock_state.py',
             'experiments/market/local_eligibility_state.py']
    prefix_errors={}
    parent=ROOT/f'experiments/results/market/curie_b10_early_request_20261013_v1_{"smoke" if args.mode=="smoke" else "fit"}.json'
    prior=json.loads(parent.read_text());inputs[str(parent.relative_to(ROOT))]=sha(parent)
    for row in rows:
        saved=next(r for r in prior['rows'] if r['arm']==row['arm'])
        error=max(abs(a-b) for a,b in zip(saved['block_nll'],row['block_nll']))
        assert error<1e-12 and saved['requests']==row['requests']
        prefix_errors[row['arm']]=error
    result = dict(source_parity=source_parity, deployment_parity=prefix_errors, original_source_wall_s=original_source['wall_s'], status='completed', battle='B10', args=vars(args), acquisition=source,
                  rows=rows, source_sha256={p: sha(ROOT/p) for p in files}, input_sha256=inputs,
                  wall_s=time.monotonic()-begin, peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  scope='Online teaching reuses the just-observed selected-proposal forecasts instead of replaying that world; actual actor parameters and pre-window numerical state must match selected proposal exactly. Only alternative world is replayed, with executed arithmetic audited. Acquisition actor uses exactBPTT and cannot reuse selected-proposal factuals: both teacher worlds remain there. Compact source preparation on unaudited frames:128float ownclock context plus retained exact actor acquisition graph; audited own/projected endpoints unchanged. Source actor/Adam/critic and deployment forecasts/requests must match saved original within1e-12. Preparation arithmetic saving requires executed ledger, not walltime. Pre-window requests from prior observed context only. Deterministic full-support source audits every2windows; online deployment audits every16windows, not a propensity-corrected random risk estimator. Frozen deployment has no audits. Common2048TRAIN exact actor acquisition; reused Aug25DEV; no TEST or efficiency/benchmark claim inferred from one run. Native private banks and current-frame counterfactual learning retained.')
    with out.open('x') as f: json.dump(result, f, indent=2); f.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ['rows', 'acquisition', 'source_sha256', 'input_sha256']}), flush=True)
    for row in rows: print(json.dumps({k: v for k, v in row.items() if k not in ['block_nll', 'requests', 'remote_computed', 'teacher_labels']}), flush=True)


if __name__ == '__main__': main()
