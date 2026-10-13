"""R1 exact finite enumeration; no model fit, corpus access or public scoring."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import resource
import time

from finite_horizon_receipt import HorizonReceipt

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    args = ap.parse_args()
    out = ROOT / f'experiments/results/credit/{args.tag}.json'
    if out.exists():
        raise FileExistsError(out)
    start = time.monotonic()
    ids = dict(objective_id='cumulative-fixed-weights', world_id='paired-world-1',
               learner_version='issued-cut-1')
    values = (-2., -1., 3., 2.5)
    predictions = [.4, -.8, 1.1, .7]
    probabilities = [.1, .2, .3, .4]
    receipt = HorizonReceipt([2, 8, 32, 128], predictions, probabilities, **ids)
    predictions[-1] = 1000.
    probabilities[-1] = 0.
    assert receipt.predictions[-1] == .7 and receipt.probabilities[-1] == .4
    targets = [receipt.corrected_target(j, values[j-1] if j else 0., value, **ids)
               for j, value in enumerate(values)]
    q = receipt.probabilities
    mean = math.fsum(p*t for p, t in zip(q, targets))
    residuals = [(values[j] - (values[j-1] if j else 0.)) -
                 (receipt.predictions[j] - (receipt.predictions[j-1] if j else 0.))
                 for j in range(len(values))]
    variance = math.fsum(p*(t-mean)**2 for p, t in zip(q, targets))
    predicted_variance = math.fsum(r*r/p for r, p in zip(residuals, q)) - math.fsum(residuals)**2
    # A nonlinear shared terminal output B(theta)=theta0^2+theta0*theta1.
    # Raw squared-error gradients must retain the same expectation; no optimizer.
    theta0, theta1 = .6, -.3
    output = theta0**2 + theta0*theta1
    jacobian = (2*theta0+theta1, theta0)
    expected_gradient = [math.fsum(p*2*(output-t)*v for p, t in zip(q, targets))
                         for v in jacobian]
    true_gradient = [2*(output-values[-1])*v for v in jacobian]
    # Predictable randomized completion, including both completion outcomes.
    survival = (.25, .5, .75, .125)
    corrected_mean = math.fsum(q[j]*(survival[j]*receipt.corrected_target(
        j, values[j-1] if j else 0., values[j], completion_probability=survival[j], **ids)
        + (1-survival[j])*receipt.predictions[-1]) for j in range(len(values)))
    naive_mean = math.fsum(q[j]*(survival[j]*targets[j] +
                          (1-survival[j])*receipt.predictions[-1]) for j in range(len(values)))
    bias = -math.fsum((1-survival[j])*residuals[j] for j in range(len(values)))
    errors = dict(terminal_mean=abs(mean-values[-1]),
                  variance=abs(variance-predicted_variance),
                  raw_gradient=max(abs(a-b) for a, b in zip(expected_gradient, true_gradient)),
                  supported_completion_mean=abs(corrected_mean-values[-1]),
                  aborted_audit_bias=abs(naive_mean-values[-1]-bias))
    assert max(errors.values()) < 1e-12, errors
    assert values[0] < 0 < values[-1] and abs(bias) > .1
    rejected = []
    invalid_calls = {
        'world': dict(ids, world_id='different-world'),
        'version': dict(ids, learner_version='new-cut'),
        'objective': dict(ids, objective_id='different-weights'),
        'zero_completion': dict(ids, completion_probability=0.),
        'excess_completion': dict(ids, completion_probability=1.1),
    }
    for name, kwargs in invalid_calls.items():
        try:
            receipt.corrected_target(0, 0., values[0], **kwargs)
        except ValueError:
            rejected.append(name)
        else:
            raise AssertionError(name)
    for name, level, lower, upper in [('anchor',0,1.,-2.),('level',4,0.,0.),
                                      ('nonfinite',0,0.,float('nan'))]:
        try:
            receipt.corrected_target(level, lower, upper, **ids)
        except ValueError:
            rejected.append(name)
        else:
            raise AssertionError(name)
    sources = ['experiments/credit/finite_horizon_receipt.py',
               'experiments/credit/check_finite_horizon_receipt.py']
    result = dict(status='completed', battle='R1', tag=args.tag, gate='PASS',
                  scope='Finite scalar receipt algebra; no native rollout or fit',
                  errors=errors, rejection_checks=rejected,
                  terminal_advantage=values[-1], early_advantage=values[0],
                  naive_completion_bias=bias, conditional_variance=variance,
                  wall_s=time.monotonic()-start,
                  max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources})
    out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
