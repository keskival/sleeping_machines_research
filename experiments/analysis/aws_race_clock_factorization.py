"""Stdlib algebra witnesses; no native forward, data scoring or training."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def verify():
    rows = []
    for rates in ((1.,), (1., 1.), (1., 2., 4.), (.01, .02, 10.)):
        total = sum(rates)
        winner = max(range(len(rates)), key=lambda k: rates[k])
        probabilities = [r / total for r in rates]
        for time in (0., .1, 1., 3.):
            # Product of per-clock survivals equals the minimum-clock survival.
            product = math.prod(math.exp(-r*time) for r in rates)
            assert math.isclose(product, math.exp(-total*time), rel_tol=1e-12, abs_tol=1e-15)
            joint = [r*math.exp(-total*time) for r in rates]
            assert all(math.isclose(j / (total*math.exp(-total*time)), p,
                                    rel_tol=1e-12) for j,p in zip(joint, probabilities))
        ratio = total / rates[winner]
        assert 1 <= ratio <= len(rates)
        rows.append(dict(rates=rates, winner_probabilities=probabilities,
                         sampled_min_mean=1/total, sampled_min_variance=1/total**2,
                         all_noise_one_delay=1/rates[winner], delay_ratio=ratio))
    # A finite output distribution isolates variance and mean bias exactly.
    probabilities = (.75, .25)
    values = (2., -2.)
    mean = sum(p*v for p,v in zip(probabilities, values))
    variance = sum(p*(v-mean)**2 for p,v in zip(probabilities, values))
    witnesses = []
    for target in (1., 2., -2.):
        sampled_risk = sum(p*(v-target)**2 for p,v in zip(probabilities, values))
        mean_risk = (mean-target)**2
        greedy_risk = (values[0]-target)**2
        assert math.isclose(sampled_risk, mean_risk + variance)
        witnesses.append(dict(target=target, sampled_squared_risk=sampled_risk,
                              mean_prediction_squared_risk=mean_risk,
                              greedy_squared_risk=greedy_risk))
    assert witnesses[0]['greedy_squared_risk'] > witnesses[0]['mean_prediction_squared_risk']
    assert witnesses[2]['greedy_squared_risk'] > witnesses[2]['sampled_squared_risk']
    # Independent increment errors in a scalar linearized closed-loop forecast.
    amplification = []
    for gain in (.5, 1., 1.1):
        horizon = 20
        direct = sum(gain**(2*k) for k in range(horizon))
        formula = horizon if gain == 1 else (gain**(2*horizon)-1)/(gain**2-1)
        assert math.isclose(direct, formula, rel_tol=1e-12)
        amplification.append(dict(gain=gain, horizon=horizon,
                                  variance_per_unit_increment_noise=direct))
    return dict(status='algebra_contracts_passed', clocks=rows,
                output_risk_witnesses=witnesses, linearized_noise_amplification=amplification,
                scope='Constructed algebra only; no native model, measured sMAPE, runtime or quality claim.',
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    assert not output.exists()
    output.write_text(json.dumps(verify(), indent=2, allow_nan=False)+'\n')
    print('Race/time factorization and risk witnesses passed; no numerical model execution.')
