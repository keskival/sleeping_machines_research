"""B1/R1 feedback-order and Monte Carlo replication invariance contracts."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import resource
import time

from packet_normalized_offset import PacketNormalizedOffset

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True); a = ap.parse_args()
    out = ROOT/'experiments/results/credit'/f'{a.tag}.json'; assert not out.exists(); start = time.monotonic()
    old, new, pred, version = {3: 1.}, {8: 1.}, {3: -.7, 8: -.9}, (2, 4, 'causal-prefix', 4)
    initial, rate = .2, .125
    samples = [(3, -.2), (8, -.3), (3, -.4)]
    target = initial+rate*(sum(value-pred[r] for r, value in samples)/len(samples)-initial)
    error = gain_error = 0.
    for copies in [1, 2, 8]:
        for order in itertools.permutations(range(3)):
            model = PacketNormalizedOffset(initial, rate)
            receipts = model.issue_batch(old, new, pred, version, 'batch', len(samples)*copies)
            planned = [receipts[j].packet.corrected_gain(r, value, receipts[j].packet.version)
                       for j, (r, value) in enumerate(samples*copies)]
            for repeat in range(copies):
                for k in order:
                    j = repeat*3+k; r, value = samples[k]
                    gain = model.consume(receipts[j], r, value, version)
                    gain_error = max(gain_error, abs(gain-planned[j]))
            error = max(error, abs(model.value-target))
            assert not model.pending
    model = PacketNormalizedOffset(initial, rate)
    one = model.issue_batch(old, new, pred, version, 'one', 1)[0]
    model.consume(one, 3, -.2, version)
    rejected = 0
    for call in [lambda: model.consume(one, 3, -.2, version),
                 lambda: model.consume(one, 3, -.2, (99,)),
                 lambda: model.issue_batch(old, new, pred, version, 'bad', 0)]:
        try: call()
        except ValueError: rejected += 1
    noop = model.issue_batch(old, old, {}, version, 'noop', 8)
    assert not model.pending and all(not r.packet.entries for r in noop)
    assert error < 1e-12 and gain_error == 0 and rejected == 3
    paths = ['experiments/credit/check_packet_normalized_offset.py', 'experiments/credit/packet_normalized_offset.py',
             'experiments/credit/policy_difference_audit.py']
    result = dict(status='completed', tag=a.tag, battle='B1/R1',
                  metrics=dict(batch_step_replication_order_error=error, issued_estimator_error=gain_error,
                               sample_repetitions=[1, 2, 8], arrival_orders=6, invalid_rejections=rejected,
                               unchanged_batch_no_pending=True),
                  whole_wall_s=time.monotonic()-start, peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  source_sha256={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},
                  scope='Exact additive scalar SGD at issued coordinates, normalized to one causal packet. Complete batches, frozen prediction/target versions; partial/missing feedback and interleaved stale-gradient stability require separate accounting. Finite algebra contracts only, no variance/utility or native online deployment claim.')
    out.write_text(json.dumps(result, indent=2)+'\n'); print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
