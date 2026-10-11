"""B1/R1 finite exhaustive witnesses for sparse policy-difference audits."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import time

from policy_difference_audit import issue

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    a = ap.parse_args()
    out = ROOT/'experiments/results/credit'/f'{a.tag}.json'
    assert not out.exists()
    start = time.monotonic()
    version = (3, 5, 7, 'issued-prefix', 4)
    prior = {1000003: .6, 9000001: .4}
    policy = {1000003: .2, 9000001: .4, 800000007: .4}
    predicted = {1000003: .7, 800000007: -.8}
    truth = {1000003: -.3, 800000007: .5}
    packet = issue(prior, policy, predicted, version)
    assert len(packet.entries) == 2  # Unchanged route needs no prediction or audit.
    proposal = dict(packet.proposal())
    exact = sum(w*truth[r] for r, w, _ in packet.entries)
    corrected = {r: packet.corrected_gain(r, truth[r], version) for r in proposal}
    mean = sum(proposal[r]*corrected[r] for r in proposal)
    variance = sum(proposal[r]*(corrected[r]-mean)**2 for r in proposal)
    residual_mean = sum(w*(truth[r]-q) for r, w, q in packet.entries)
    variance_formula = packet.l1_distance**2*sum(proposal[r]*(truth[r]-q)**2 for r, _, q in packet.entries)-residual_mean**2
    mean_error, variance_error = abs(mean-exact), abs(variance-variance_formula)
    shifted = issue(prior, policy, {r: q+100 for r, q in predicted.items()}, version)
    shift_error = max(abs(shifted.corrected_gain(r, truth[r]+100, version)-corrected[r]) for r in proposal)
    empty = issue(prior, prior, {}, version)
    assert not empty.entries and not empty.proposal() and empty.predicted_gain == 0
    rejected = 0
    for action in [lambda: packet.corrected_gain(1000003, -.3, (4, 5, 7, 'issued-prefix', 4)),
                   lambda: packet.corrected_gain(9000001, 0., version),
                   lambda: issue(prior, policy, {}, version),
                   lambda: issue({1: .2}, {1: 1.}, {}, version)]:
        try: action()
        except ValueError: rejected += 1
    epsilon = max(abs(truth[r]-q) for r, _, q in packet.entries)
    bound = epsilon*packet.l1_distance
    bound_slack = bound-abs(packet.predicted_gain-exact)
    assert mean_error < 1e-12 and variance_error < 1e-12 and shift_error < 1e-12
    assert rejected == 4 and bound_slack >= -1e-12
    sources = ['experiments/credit/policy_difference_audit.py', 'experiments/credit/check_policy_difference_audit.py']
    result = dict(status='completed', tag=a.tag, battle='B1/R1',
                  metrics=dict(exhaustive_mean_error=mean_error, exhaustive_variance_error=variance_error,
                               common_offset_error=shift_error, invalid_rejections=rejected,
                               changed_support=len(packet.entries), sparse_union=3, unchanged_requires_no_prediction=True,
                               noop_comparison_requires_no_audit=True, gain_error_bound_slack=bound_slack),
                  whole_wall_s=time.monotonic()-start, peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  source_sha256={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},
                  scope='Finite exhaustive algebra, sparse map implementation; unbiasedness requires frozen issued policies and conditionally unbiased future cost observations. No fitted critic/actor, measured hardware scaling or utility win. Candidate discovery and issued predictions remain additional paid work.')
    out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
