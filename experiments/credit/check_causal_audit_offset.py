"""B1/R1 audit centering algebra, delayed feedback and native world-list parity."""
import argparse
import hashlib
import json
import resource
import time

import torch
from torch.utils.flop_counter import FlopCounterMode

from causal_audit_offset import CausalOffset
from paired_native_replay import replay_pairs
from optimizer_aware_route_credit import replay, ROOT, native


def variance(packet, truth):
    mean = sum(p*packet.corrected_gain(a, truth[a], packet.version) for a, p in packet.proposal())
    return mean, sum(p*(packet.corrected_gain(a, truth[a], packet.version)-mean)**2 for a, p in packet.proposal())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True); a = ap.parse_args()
    torch.set_default_dtype(torch.float64); torch.set_num_threads(1)
    out = ROOT/'experiments/results/credit'/f'{a.tag}.json'; assert not out.exists()
    start = time.monotonic()
    prior, policy, pred, truth, version = {2: 1.}, {9: 1.}, {2: -.7, 9: -.9}, {2: -.2, 9: -.3}, (4, 5, 'prefix', 4)
    cstar = sum(truth[k]-pred[k] for k in pred)/2
    best = CausalOffset(cstar).issue(prior, policy, pred, version, 'best')
    bm, bv = variance(best.packet, truth)
    formula_error = mean_error = 0.
    for offset in [0., .2, .8, -1.]:
        receipt = CausalOffset(offset).issue(prior, policy, pred, version, offset)
        mean, var = variance(receipt.packet, truth)
        mean_error = max(mean_error, abs(mean-bm))
        formula_error = max(formula_error, abs(var-bv-4*(offset-cstar)**2))
        assert abs(receipt.packet.predicted_gain-best.packet.predicted_gain) < 1e-12
    causal = CausalOffset(.2)
    first = causal.issue(prior, policy, pred, version, 1)
    later = causal.issue(prior, policy, pred, version, 2)
    issued_first = first.packet.corrected_gain(2, truth[2], first.packet.version)
    issued_later = later.packet.corrected_gain(9, truth[9], later.packet.version)
    value0 = causal.value
    # Out-of-order feedback is valid, but the issued estimator cannot change.
    causal.consume(later, 9, truth[9], version)
    expected1 = value0+.125*(truth[9]-pred[9]-value0)
    update_error = abs(causal.value-expected1)
    actual_first = causal.consume(first, 2, truth[2], version)
    assert actual_first == issued_first and issued_later == later.packet.corrected_gain(9, truth[9], later.packet.version)
    rejected = 0
    for call in [lambda: causal.consume(first, 2, truth[2], version),
                 lambda: causal.consume(later, 9, truth[9], (99,)),
                 lambda: CausalOffset(float('nan'))]:
        try: call()
        except ValueError: rejected += 1
    assert rejected == 3 and not causal.pending
    empty = causal.issue(prior, prior, {}, version, 'unchanged')
    assert not empty.packet.entries and not causal.pending
    actor = native.actor(4, 170)
    t, m = native.base.batch(512345, size=1, pairs=2)
    actions = native.base.actions()
    # Arbitrary list, including different routes, nonzero messages and one no-op.
    routes, ids = [4, 7, 4], [3, 8, 0]
    with FlopCounterMode(display=False) as dense_work:
        dense, dense_head, _ = replay(actor, t, m, list(range(8)), actions, .05)
    with FlopCounterMode(display=False) as sparse_work:
        sparse, sparse_head = replay_pairs(actor, t, m, routes, actions[ids], .05)
    index = torch.tensor([r*9+i for r, i in zip(routes, ids)])
    score_error = float((dense[index]-sparse).abs().max())
    weight_error = max(float((dense_head[k][index]-v).abs().max()) for k, v in sparse_head.items())
    assert formula_error < 1e-12 and mean_error < 1e-12 and update_error < 1e-12
    assert score_error < 1e-11 and weight_error < 1e-11
    paths = ['experiments/credit/check_causal_audit_offset.py', 'experiments/credit/causal_audit_offset.py',
             'experiments/credit/policy_difference_audit.py', 'experiments/credit/paired_native_replay.py',
             'experiments/credit/optimizer_aware_route_credit.py', *native.SOURCES]
    result = dict(status='completed', battle='B1/R1', tag=a.tag,
                  metrics=dict(centering_variance_identity_error=formula_error, centered_mean_error=mean_error,
                               causal_update_error=update_error, invalid_rejections=rejected,
                               delayed_packet_estimator_unchanged=True, noop_no_pending=True,
                               native_list_score_error=score_error, native_list_adam_parameter_error=weight_error,
                               dense_worlds=72, list_worlds=3, dense_supported_flops=dense_work.get_total_flops(),
                               list_supported_flops=sparse_work.get_total_flops()),
                  whole_wall_s=time.monotonic()-start, peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  source_sha256={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},
                  scope='Finite centering/delay/version witnesses and same-incoming-native-state three-world Adam parity. Dense72 versus list3 work measures this replay kernel only, with different output coverage; no full-gradient/fit efficiency or learned calibration utility claim.')
    out.write_text(json.dumps(result, indent=2)+'\n'); print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
