"""B1/R1 mathematical bridge from pairwise cause credit to causal memory writes.

No fitting. The small exponential-clock case is an exact member diagnostic,
not an integrated-model quality result. CLI execution belongs to a battle queue.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def posterior(scores):
    weights = [math.exp(s - max(scores)) for s in scores]
    total = sum(weights)
    return [w / total for w in weights]


def cell_score(state, weights, biases, gap, cell, mark):
    """Log probability of one gap recording cell and mark, with constant clocks."""
    logits = [w * state + b for w, b in zip(weights, biases)]
    total = sum(math.exp(x) for x in logits)
    return -total * gap + math.log(-math.expm1(-total * cell)) + logits[mark] - math.log(total)


def cell_adjoint(state, weights, biases, gap, cell, mark):
    rates = [math.exp(w * state + b) for w, b in zip(weights, biases)]
    total = sum(rates); derivative = sum(w * r for w, r in zip(weights, rates))
    return weights[mark] - derivative / total + derivative * (cell / math.expm1(total * cell) - gap)


def transition(scores, proposal):
    p = posterior(scores); size = len(p); kernel = [[0.0] * size for _ in p]
    for i in range(size):
        for j in range(size):
            if i != j:
                kernel[i][j] = proposal[j] * min(1.0, p[j] * proposal[i] / (p[i] * proposal[j]))
        kernel[i][i] = 1.0 - sum(kernel[i])
    return kernel


def checks():
    weights, biases = [1.2, -.7], [.1, -.2]
    gap, cell, mark = .7, .125, 1
    state = .3; step = 1e-6
    adjoint = cell_adjoint(state, weights, biases, gap, cell, mark)
    finite_difference = (cell_score(state + step, weights, biases, gap, cell, mark) -
                         cell_score(state - step, weights, biases, gap, cell, mark)) / (2 * step)
    derivative_error = abs(adjoint - finite_difference)
    assert derivative_error < 1e-8
    # Normalize over every mark and recording cell. The residual tail is known.
    rates = [math.exp(w * state + b) for w, b in zip(weights, biases)]
    n_cells = 160
    mass = sum(math.exp(cell_score(state, weights, biases, k * cell, cell, m))
               for k in range(n_cells) for m in range(len(weights)))
    normalization_error = abs(mass + math.exp(-sum(rates) * n_cells * cell) - 1)
    assert normalization_error < 1e-12
    prior = [.15, .25, .6]; writes = [-.8, .1, .9]; elapsed, decay = .7, .4
    retention = math.exp(-decay * elapsed)
    states = [retention * (state + v) for v in writes]
    exact = [math.log(pi) + cell_score(s, weights, biases, gap, cell, mark)
             for pi, s in zip(prior, states)]
    base_state = retention * state
    derivative = cell_adjoint(base_state, weights, biases, gap, cell, mark)
    surrogate = [math.log(pi) + cell_score(base_state, weights, biases, gap, cell, mark) +
                 derivative * retention * v for pi, v in zip(prior, writes)]
    exact_p, approx_p = posterior(exact), posterior(surrogate)
    residuals = [s - t for s, t in zip(exact, surrogate)]
    # Common shifts cancel; the half-span is the tight score-error certificate.
    epsilon = (max(residuals) - min(residuals)) / 2
    total_variation = sum(abs(a - b) for a, b in zip(exact_p, approx_p)) / 2
    assert total_variation <= math.tanh(epsilon / 2) + 1e-14
    proposal = [.05, .8, .15]
    kernel = transition(exact, proposal)
    stationary_error = max(abs(sum(exact_p[i] * kernel[i][j] for i in range(3)) - exact_p[j]) for j in range(3))
    assert stationary_error < 1e-14
    # Correcting against the adjoint approximation preserves the wrong posterior.
    approximate_kernel = transition(surrogate, proposal)
    bias = sum(abs(sum(exact_p[i] * approximate_kernel[i][j] for i in range(3)) - exact_p[j]) for j in range(3)) / 2
    assert bias > 1e-3
    # Two write variables, two observations: local next-event odds differ from full-horizon odds.
    prior2, writes2 = [.4, .6], [-.6, .7]
    full = []
    for pi, s in zip(prior, states):
        continuation = sum(pj * math.exp(cell_score(retention * (s + v), weights, biases, .4, cell, 0))
                           for pj, v in zip(prior2, writes2))
        full.append(math.log(pi) + cell_score(s, weights, biases, gap, cell, mark) + math.log(continuation))
    full_p = posterior(full)
    horizon_tv = sum(abs(a - b) for a, b in zip(exact_p, full_p)) / 2
    assert horizon_tv > 1e-3
    # For two-route opposing score errors the TV bound can be attained exactly.
    extremal_error = 0.0
    for e in (.001, .1, .5, 2.0):
        a = posterior([-e / 2, e / 2]); b = posterior([e / 2, -e / 2])
        extremal_error = max(extremal_error, abs(sum(abs(x-y) for x,y in zip(a,b))/2 - math.tanh(e / 2)))
    assert extremal_error < 1e-14
    return dict(cell_normalization_error=normalization_error, cell_adjoint_finite_difference_error=derivative_error,
                exact_hidden_write_mh_stationary_error=stationary_error,
                adjoint_score_half_span=epsilon, adjoint_posterior_tv=total_variation,
                posterior_tv_bound=math.tanh(epsilon / 2), approximate_mh_exact_posterior_drift=bias,
                next_event_vs_two_event_posterior_tv=horizon_tv, sharp_bound_error=extremal_error,
                scope='fixed causal prefix, one observed-event-time write; no fitting, no complete hidden-history claim')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--tag', required=True); args = parser.parse_args()
    me = Path(__file__).resolve()
    result = dict(status='completed', battle='B1/R1 route-credit integration', tag=args.tag, checks=checks(),
                  source_sha256={str(me.relative_to(ROOT)): hashlib.sha256(me.read_bytes()).hexdigest()})
    with (ROOT / 'experiments/results/credit' / f'{args.tag}.json').open('x') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    print(json.dumps(result['checks'], indent=2))
