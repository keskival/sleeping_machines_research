"""B1/R1 paired alternative observation, no-op cancellation and variance checks."""
import argparse
import hashlib
import json
import resource
import time

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

from measure_causal_audit_offset import measure, ROOT, SOURCES as BASE_SOURCES
from paired_native_replay import replay_pairs
from optimizer_aware_route_credit import native

TAG = 'curie_paired_audit_20261011T0740Z'
SOURCES = ['experiments/credit/paired_policy_audit.py', *BASE_SOURCES]


def contract():
    actor = native.actor(4, 170)
    t, m = native.base.batch(623456, size=1, pairs=2)
    actions = native.base.actions()
    three, state3 = replay_pairs(actor, t, m, [4, 7, 0], actions[[3, 8, 0]], .05)
    two, state2 = replay_pairs(actor, t, m, [4, 7], actions[[3, 8]], .05)
    score_error = float((two-three[:2]).abs().max())
    adam_error = max(float((state2[k]-v[:2]).abs().max()) for k, v in state3.items())
    difference_error = float(((three[0]-three[2])-(three[1]-three[2])-(two[0]-two[1])).abs().max())
    work = []
    for name, routes, ids in [('paired_alternatives', [4, 7], [3, 8]), ('sampled_plus_noop', [4, 0], [3, 0])]:
        tick = time.monotonic()
        with FlopCounterMode(display=False) as counter:
            for _ in range(10): replay_pairs(actor, t, m, routes, actions[ids], .05)
        work.append(dict(arm=name, worlds=2, iterations=10, supported_flops=counter.get_total_flops(), instrumented_wall_s=time.monotonic()-tick))
    x = np.array([-.3, .2, .5, -.1]); y = np.array([.4, -.2, .1, .3])
    dominance_slack = float('inf'); identity_error = 0.
    pairvar = float(np.var(x-y)); S = float(np.var(x)+np.var(y))
    for predicted_sum in [-1., 0., float(x.mean()+y.mean()), 2.]:
        sampled = np.concatenate((2*x-predicted_sum, predicted_sum-2*y))
        observedvar = float(np.var(sampled))
        formula = 2*S+float(x.mean()+y.mean()-predicted_sum)**2
        identity_error = max(identity_error, abs(observedvar-formula))
        dominance_slack = min(dominance_slack, observedvar-pairvar)
    assert score_error < 1e-11 and adam_error < 1e-11 and difference_error < 1e-11
    assert identity_error < 1e-12 and dominance_slack >= -1e-12
    assert work[0]['supported_flops'] == work[1]['supported_flops']
    return dict(native_score_error=score_error, native_adam_error=adam_error, noop_cancellation_error=difference_error,
                single_variance_identity_error=identity_error, min_dominance_slack=dominance_slack, work=work)


def paired_measure(smoke):
    metrics, payload = measure(smoke, 774 if smoke else 475, 594 if smoke else 495)
    records = payload['records']; pairvars = []; zero = []; best = []; worst_slack = float('inf')
    for row in records:
        if row['old'] == row['new']:
            pairvars.append(0.); zero.append(0.); best.append(0.); continue
        old = np.array([o[row['old']] for o in row['observed']])
        new = np.array([o[row['new']] for o in row['observed']])
        variance = float(np.var(old-new)); S = float(np.var(old)+np.var(new))
        worst_slack = min(worst_slack, 2*S-variance)
        assert variance <= 2*S+1e-11
        pairvars.append(variance); zero.append(row['variance']['zero_predictor'])
        best.append(min(row['variance'].values()))
        row['paired_variance'] = variance; row['optimal_single_variance'] = 2*S
    pairvars, zero, best = np.array(pairvars), np.array(zero), np.array(best)
    boot = np.random.default_rng(595).integers(0, len(records), (4000, len(records)))
    metrics['paired'] = dict(mean_variance=float(pairvars.mean()), mean_zero_single_variance=float(zero.mean()),
                             ratio_vs_zero_single=float(pairvars.mean()/zero.mean()) if zero.mean() else None,
                             ratio_vs_best_single_per_prefix=float(pairvars.mean()/best.mean()) if best.mean() else None,
                             variance_saving_bootstrap95=np.quantile((zero-pairvars)[boot].mean(1), [.025, .975]).tolist(),
                             min_optimal_single_dominance_slack=None if not np.isfinite(worst_slack) else worst_slack,
                             operational_worlds=metrics['operational_sampled_worlds'])
    return metrics, payload


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True)
    ap.add_argument('--mode', choices=['contract', 'smoke', 'measure'], required=True); args = ap.parse_args()
    torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
    out = ROOT/'experiments/results/credit'/f'{args.tag}.json'; assert not out.exists(); start = time.monotonic()
    if args.mode != 'contract':
        required = ['contract']+(['smoke'] if args.mode == 'measure' else [])
        for mode in required:
            d = json.loads((ROOT/f'experiments/results/credit/{TAG}_{mode}.json').read_text()); assert d['status'] == 'completed'
            for p, digest in d['source_sha256'].items(): assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest() == digest, p
    if args.mode == 'contract': metrics, payload = contract(), None
    else: metrics, payload = paired_measure(args.mode == 'smoke')
    result = dict(status='completed', tag=args.tag, mode=args.mode, battle='B1/R1', metrics=metrics,
                  whole_wall_s=time.monotonic()-start, peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  source_sha256={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCES},
                  scope='Same fixed native depth4 actor/proposal; four causal branch-local Adam updates. Paired operational audit replays two alternatives; single-route audit replays one alternative plus no-op. Equal two-world kernel FLOPs checked. New475 finite future stream, no official TEST; three diagnostic worlds plus legacy calibration/control processing charged. Paired variance includes individual future noise; conditional algebraic dominance does not imply whole-gradient/training/depth or model-quality win. Existing fits/discovery costs remain additional.')
    if payload is not None:
        cp = out.with_suffix('.pt'); torch.save(payload, cp)
        result.update(checkpoint=str(cp.relative_to(ROOT)), checkpoint_sha256=hashlib.sha256(cp.read_bytes()).hexdigest())
    out.write_text(json.dumps(result, indent=2)+'\n'); print(json.dumps(metrics, indent=2))


if __name__ == '__main__': main()
