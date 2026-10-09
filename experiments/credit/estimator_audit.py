"""Exact finite-cause audit of note 160's self-normalized credit estimator.
No fitting, torch, sampling, or benchmark execution. Enumerates multinomial counts.
"""
import itertools
import math


def compositions(total, n):
    if n == 1:
        yield (total,)
    else:
        for first in range(total + 1):
            for rest in compositions(total - first, n - 1):
                yield (first,) + rest


def expected_credit(rho, q, proposals):
    """E[rho_hat] for iid replacement proposals; exact enumeration, float64."""
    out = [0.0] * len(q)
    for counts in compositions(proposals, len(q)):
        probability = math.factorial(proposals)
        for count, prob in zip(counts, q):
            probability *= prob ** count / math.factorial(count)
        weights = [count * target / prop for count, target, prop in zip(counts, rho, q)]
        normalizer = sum(weights)
        for i, weight in enumerate(weights):
            out[i] += probability * weight / normalizer
    return out


def linear_mac_estimate(depth, hidden=64, proposals=2, d=16, K=8, K2=4, C=10):
    """v5's leading-linear training MAC convention; excludes nonlinear/Adam work."""
    experts = K * (K2 if depth == 2 else 1)
    router = d * K + (d * K * K2 if depth == 2 else 0)
    expert = C * d
    credit = d * hidden + hidden * hidden + hidden * experts
    dense = 3 * (router + experts * expert)
    closed = 3 * router + 3 * proposals * expert + router + expert + 6 * credit
    return dict(depth=depth, credit_hidden=hidden, dense_linear_macs=dense,
                closed_linear_macs=closed, closed_over_dense=closed / dense)


def audit():
    rho, q = [0.8, 0.2], [0.2, 0.8]
    one = expected_credit(rho, q, 1)
    assert max(abs(a-b) for a, b in zip(one, q)) < 1e-14
    rows = []
    for s in (1, 2, 4, 8, 16, 32):
        expected = expected_credit(rho, q, s)
        rows.append(dict(proposals=s, expected_credit=expected,
                         posterior_tv_bias=0.5 * sum(abs(a-b) for a,b in zip(expected,rho)),
                         expected_wake_q_logit_update=[a-b for a,b in zip(expected,q)]))
    assert rows[1]['posterior_tv_bias'] > 0.3
    for s in (1, 2, 8):
        matched = expected_credit(rho, rho, s)
        assert max(abs(a-b) for a,b in zip(matched,rho)) < 1e-13
    # Local feedback strength at the posterior, in probability coordinates:
    # d(E[rho_hat] - q)/dq = -(1 - 1/S) on the simplex tangent.
    derivatives = []
    eps = 1e-6
    for s in (1, 2, 4, 8):
        plus = expected_credit(rho, [rho[0]+eps, rho[1]-eps], s)[0] - (rho[0]+eps)
        minus = expected_credit(rho, [rho[0]-eps, rho[1]+eps], s)[0] - (rho[0]-eps)
        observed = (plus-minus)/(2*eps)
        assert abs(observed + (1-1/s)) < 1e-7
        derivatives.append(dict(proposals=s, wake_feedback_derivative=observed))
    # Enumerating each cause ONCE is a uniform proposal contract, not a
    # nonuniform iid proposal contract: dividing by q in that case is wrong.
    weights = [a/b for a,b in zip(rho,q)]
    wrong = [w/sum(weights) for w in weights]
    assert abs(wrong[0] - rho[0]) > 0.1
    return dict(status='completed', scope='exact finite-cause mathematical audit; no fitting',
                finite_proposal_bias=rows, local_feedback=derivatives,
                nonuniform_once_each_credit=wrong,
                modeled_linear_work=[linear_mac_estimate(depth,h) for depth,h in itertools.product((1,2),(64,16,8))])


if __name__ == '__main__':
    import json
    print(json.dumps(audit(), indent=2))
