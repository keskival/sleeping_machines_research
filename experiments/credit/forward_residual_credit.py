"""B1/R1 finite mathematics for forward-supported paired residual credit.

No fitting or benchmark access. Targets are the conditional expected-loss
route component, not a normalized marginal-likelihood posterior estimator.
"""
import math


def add(a, b):
    return [x + y for x, y in zip(a, b)]


def scale(a, value):
    return [value * x for x in a]


def norm2(a):
    return sum(x * x for x in a)


def score_vectors(probabilities, jacobian):
    """Pull softmax log-probability scores through a shared parameter Jacobian."""
    mean = [sum(p * row[j] for p, row in zip(probabilities, jacobian))
            for j in range(len(jacobian[0]))]
    return [[x - y for x, y in zip(row, mean)] for row in jacobian]


def full_credit(probabilities, utilities, scores):
    return [sum(p * f * h[j] for p, f, h in zip(probabilities, utilities, scores))
            for j in range(len(scores[0]))]


def paired_credit(probabilities, utilities, predictions, scores, winner, other, propensity):
    if other == winner or propensity <= 0:
        raise ValueError('A supported nonwinner audit is required')
    baseline = full_credit(probabilities, predictions, scores)
    residual = (utilities[other] - utilities[winner]) - (predictions[other] - predictions[winner])
    return add(baseline, scale(scores[other], probabilities[other] * residual / propensity))


def enumerate_pairs(probabilities, utilities, predictions, scores, proposals):
    target = full_credit(probabilities, utilities, scores)
    mean = [0.] * len(target); variance = 0.
    for winner, p in enumerate(probabilities):
        q = proposals[winner]
        if q[winner] != 0 or abs(sum(q) - 1) > 1e-12:
            raise ValueError('Conditional nonwinner proposal must normalize')
        for other, chance in enumerate(q):
            if other == winner:
                continue
            if chance <= 0:
                raise ValueError('Full alternative support is required')
            estimate = paired_credit(probabilities, utilities, predictions, scores, winner, other, chance)
            weight = p * chance
            mean = add(mean, scale(estimate, weight))
            variance += weight * norm2(add(estimate, scale(target, -1)))
    return dict(mean=mean, target=target, variance_trace=variance,
                maximum_mean_error=max(abs(a-b) for a,b in zip(mean,target)))


def allocation(magnitudes, costs, budget, floor=.01):
    """Optimal independent audit probabilities for a fixed quadratic bound.

    magnitudes may be certified residual bounds or oracle residual norms for
    a mathematical check. Learned estimates do not certify zero residuals.
    """
    if len(magnitudes) != len(costs) or not 0 < floor <= 1 or any(c <= 0 for c in costs):
        raise ValueError('Positive costs and supported audit floor required')
    if any(a < 0 for a in magnitudes) or budget < floor * sum(costs):
        raise ValueError('Budget cannot satisfy the audit support floor')
    if budget >= sum(costs):
        return [1.] * len(costs)

    def probabilities(multiplier):
        return [max(floor, min(1., a / math.sqrt(multiplier * c))) for a,c in zip(magnitudes,costs)]

    high = 1.
    while sum(p*c for p,c in zip(probabilities(high),costs)) > budget:
        high *= 4
    low = 0.
    for _ in range(100):
        mid = (low+high)/2
        if sum(p*c for p,c in zip(probabilities(mid),costs)) > budget:
            low = mid
        else:
            high = mid
    return probabilities(high)
