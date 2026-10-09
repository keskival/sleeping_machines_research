"""B1/R1 non-fitting audit: posterior credit for causal hidden write histories.

Two sparse writes, state-dependent route priors, decaying persistent memory,
and marked exponential races. Exhaustive enumeration is a tiny-model oracle.
"""
import itertools
import math


GAPS = (0.7, 1.3)
MARKS = (0, 1)
WRITES = (-0.9, 1.2)


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def trajectory(theta, routes, gradients=False):
    z = dz = 0.0
    score = grad = 0.0
    for gap, mark, route in zip(GAPS, MARKS, routes):
        # Prediction before observing/updating: event plus preceding silence.
        rates = (math.exp(0.4*z), math.exp(-0.6*z))
        score += math.log(rates[mark]) - gap*sum(rates)
        grad += ((0.4, -0.6)[mark] - gap*(0.4*rates[0]-0.6*rates[1]))*dz
        # Hidden write sampled after this observed event; prior is causal.
        p = sigmoid(theta + 0.3*z)
        score += math.log(p if route else 1-p)
        grad += (route-p)*(1+0.3*dz)
        decay = math.exp(-0.2*gap)
        z = decay*z + theta*WRITES[route]
        dz = decay*dz + WRITES[route]
    # A future query with a different gap supplies credit to both writes.
    rates = (math.exp(0.4*z), math.exp(-0.6*z))
    score += math.log(rates[1]) - 0.9*sum(rates)
    grad += (-0.6 - 0.9*(0.4*rates[0]-0.6*rates[1]))*dz
    return (score, grad) if gradients else score


def marginal(theta):
    scores = [trajectory(theta, c) for c in itertools.product((0, 1), repeat=2)]
    maximum = max(scores)
    return maximum + math.log(sum(math.exp(s-maximum) for s in scores))


def audit():
    rows = []
    for theta in (-0.7, 0.2, 0.9):
        causes = list(itertools.product((0, 1), repeat=2))
        values = [trajectory(theta, c, True) for c in causes]
        evidence = marginal(theta)
        posterior = [math.exp(s-evidence) for s, _ in values]
        exact = sum(p*g for p, (_, g) in zip(posterior, values))
        eps = 1e-5
        fd = (marginal(theta+eps)-marginal(theta-eps))/(2*eps)
        trajectory_error = max(abs(g-(trajectory(theta+eps,c)-trajectory(theta-eps,c))/(2*eps))
                               for c, (_,g) in zip(causes, values))
        # Block replacement must replay downstream route priors AND emissions.
        old, new = (0, 0), (1, 0)
        full_odds = trajectory(theta,new)-trajectory(theta,old)
        prefix_only_odds = theta  # first write's prior log odds only
        assert abs(full_odds-prefix_only_odds) > 0.01
        assert abs(exact-fd) < 1e-8 and trajectory_error < 1e-8
        rows.append(dict(theta=theta, posterior=posterior, posterior_mean_score=exact,
                         marginal_finite_difference=fd, gradient_error=abs(exact-fd),
                         eligibility_error=trajectory_error, full_replayed_log_odds=full_odds,
                         prefix_only_log_odds=prefix_only_odds))
    return dict(status='completed', battle='B1/R1 hidden-write credit integration',
                scope='stdlib exhaustive mathematical audit; no fitting or benchmark score',
                causes=[list(c) for c in itertools.product((0,1),repeat=2)], rows=rows,
                decision='Complete-data hidden-write score needs causal suffix replay and memory sensitivities; prefix-only odds fail')


if __name__ == '__main__':
    import json
    print(json.dumps(audit(), indent=2))
