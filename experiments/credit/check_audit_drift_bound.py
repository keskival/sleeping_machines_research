"""B1/R1 finite tree witnesses for causal calibration with noise-coupled drift."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import resource
import time

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True); args = ap.parse_args()
    out = ROOT/'experiments/results/credit'/f'{args.tag}.json'; assert not out.exists()
    start = time.monotonic(); rows = []
    for eta, sigma, delta, initial in [(.125, .02, .001, .05), (.75, .02, .002, 0.), (1., .02, .002, .1), (.125, .02, 0., 0.)]:
        B = delta/eta+sigma*math.sqrt(eta/(2-eta))
        errors = [initial]; previous_rms = abs(initial)
        minimum_slack = float('inf'); stationary_error = 0.
        for step in range(1, 9):
            updated = []
            for old in errors:
                for noise in [-sigma, sigma]:
                    before_drift = (1-eta)*old+eta*noise
                    # Adversarial drift depends on the newly observed noise; no independence shortcut.
                    drift = -math.copysign(delta, before_drift)
                    updated.append(before_drift-drift)
            errors = updated
            rms = math.sqrt(sum(e*e for e in errors)/len(errors))
            recursive = math.sqrt((1-eta)**2*previous_rms**2+eta**2*sigma**2)+delta
            bound = B+(1-eta)**step*max(abs(initial)-B, 0.)
            minimum_slack = min(minimum_slack, recursive-rms, bound-rms)
            if delta == 0 and initial == 0:
                exact = eta*sigma*sigma/(2-eta)*(1-(1-eta)**(2*step))
                stationary_error = max(stationary_error, abs(rms*rms-exact))
            assert rms <= recursive+1e-12 and rms <= bound+1e-12
            previous_rms = rms
        rows.append(dict(eta=eta, sigma=sigma, delta=delta, initial=initial, noise_paths=len(errors),
                         final_rms=previous_rms, steady_bound=B, minimum_bound_slack=minimum_slack,
                         stationary_mean_square_identity_error=stationary_error))
    result = dict(status='completed', battle='B1/R1', tag=args.tag, rows=rows,
                  whole_wall_s=time.monotonic()-start, peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  source_sha256={str(Path(__file__).relative_to(ROOT)): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
                  scope='Eight-step finite noise trees including adversarial noise-dependent drift and eta1 edge case. Conditional drift/noise constants are assumptions, not estimates for the native pilot; no stationarity, real-domain performance or efficiency claim.')
    out.write_text(json.dumps(result, indent=2)+'\n'); print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
