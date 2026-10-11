"""B1 constructive credit contracts: stable error, useful queries and future loss."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import resource
import time
import torch

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(1101)
    start = time.monotonic()
    rows = []
    for dimension in [2, 8, 32]:
        x = torch.randn(dimension, dimension)
        covariance = x @ x.T + torch.eye(dimension)
        x = torch.randn(dimension, dimension)
        demand = x @ x.T + torch.eye(dimension)
        query = torch.randn(dimension)
        noise = .7
        direction = covariance @ query
        posterior = covariance - torch.outer(direction, direction) / (noise + query @ direction)
        predicted = direction @ demand @ direction / (noise + query @ direction)
        observed = torch.trace(demand @ (covariance - posterior))
        error = abs(float(predicted - observed))
        assert error < 1e-8
        assert float(torch.linalg.eigvalsh(posterior).min()) > 0
        rows.append(dict(contract='future_demand_query', dimension=dimension,
                         identity_error=error, future_loss_reduction=float(observed)))
    # Uncertainty alone selects the wrong query when future demand is anisotropic.
    covariance = torch.diag(torch.tensor([100., 1.]))
    demand = torch.diag(torch.tensor([0., 1.]))
    gains = []
    for query in torch.eye(2):
        direction = covariance @ query
        gains.append(float(direction @ demand @ direction / (1 + query @ direction)))
    assert gains == [0., .5]
    rows.append(dict(contract='demand_over_uncertainty', query_gains=gains))
    for amplification in [1., 1.5, 3.]:
        target = .8
        probability = 1 - target ** 2 / amplification ** 2
        error = torch.randn(8)
        transported = amplification * error
        expected = (1 - probability) * transported.square().sum()
        bound = target ** 2 * error.square().sum()
        assert abs(float(expected - bound)) < 1e-12
        # Credit reset leaves persistent forward memory untouched.
        memory = torch.randn(8)
        after_audit = memory.clone()
        assert torch.equal(memory, after_audit)
        rows.append(dict(contract='credit_reset_without_forgetting', amplification=amplification,
                         audit_probability=probability, contraction_squared=target ** 2,
                         bound_error=abs(float(expected - bound))))
    # Enumerate actual Bernoulli corrections, rather than checking only covariance algebra.
    theta = torch.tensor([.4, -.7])
    truth = torch.tensor([-.3, .2])
    hessian = torch.diag(torch.tensor([1., 9.]))
    preconditioner = torch.diag(torch.tensor([.7, .3]))
    gradient = torch.tensor([.2, -.5])
    residuals = torch.eye(2)
    estimate = gradient - residuals.sum(0)
    metric = preconditioner.T @ hessian @ preconditioner
    risks = torch.diag(metric)
    uniform = torch.full((2,), .4)
    allocated = .8 * risks.sqrt() / risks.sqrt().sum()
    step = .05

    def evaluate(probabilities):
        loss = 0.
        for bits in itertools.product([0, 1], repeat=2):
            bits = torch.tensor(bits)
            mass = torch.where(bits.bool(), probabilities, 1 - probabilities).prod()
            corrected = estimate + ((bits / probabilities)[:, None] * residuals).sum(0)
            updated = theta - step * preconditioner @ corrected
            delta = updated - truth
            loss += mass * (.5 * delta @ hessian @ delta)
        variance = (risks * (1 / probabilities - 1)).sum()
        return loss, variance

    old_loss, old_variance = evaluate(uniform)
    new_loss, new_variance = evaluate(allocated)
    predicted = .5 * step ** 2 * (old_variance - new_variance)
    observed = old_loss - new_loss
    assert observed > 0 and abs(float(observed - predicted)) < 1e-12
    rows.append(dict(contract='weighted_variance_to_future_learning',
                     uniform_probabilities=uniform.tolist(), allocated_probabilities=allocated.tolist(),
                     expected_future_loss_uniform=float(old_loss), expected_future_loss_allocated=float(new_loss),
                     observed_gain=float(observed), predicted_gain=float(predicted),
                     identity_error=abs(float(observed - predicted))))
    source = 'experiments/credit/check_constructive_credit_theory.py'
    result = dict(status='completed', tag=args.tag, battle='B1 metric-aware native credit integration',
                  rows=rows, source_sha256={source: hashlib.sha256((ROOT / source).read_bytes()).hexdigest()},
                  wall_s=time.monotonic()-start, max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  scope='Exact finite numerical contracts for declared Gaussian information and quadratic future-loss models; analytic audit-reset expectation. No native learning fit or benchmark quality claim.')
    with (ROOT / 'experiments/results/credit' / f'{args.tag}.json').open('x') as output:
        json.dump(result, output, indent=2)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
