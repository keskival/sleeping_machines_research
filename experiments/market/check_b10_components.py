"""B10 numerical observer contract; no fitting or benchmark test access."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from b10_tpp import Poisson, Hawkes, Race
from b10_components import components

ROOT = Path(__file__).resolve().parents[2]


def checks():
    torch.set_num_threads(1); torch.set_default_dtype(torch.float64); torch.manual_seed(43)
    t = torch.tensor([[0, 0, 1e-6, .01, .02, .02, .05, .1, .2],
                      [0, 1e-6, 1e-6, .005, .01, .03, .03, .11, .3]])
    marks = torch.tensor([[0, 1, 1, 3, 4, 2, 5, 0, 1], [3, 3, 5, 2, 0, 1, 4, 3, 5]])
    models = dict(poisson=Poisson([1.] * 6), hawkes=Hawkes([1.] * 6),
                  race=Race(np.array([1e-6, .001, .01, .1]), 8, 2))
    rows = {}
    with torch.no_grad():
        for name, model in models.items():
            model.eval()
            timing, mark = components(model, t, marks)
            error = (timing + mark - model.terms(t, marks)).abs().max().item()
            assert error < 1e-10 and torch.isfinite(timing).all() and torch.isfinite(mark).all()
            assert (timing <= 1e-10).all() and (mark <= 1e-10).all()
            altered = marks.clone(); altered[:, 7] = (altered[:, 7] + 1) % 6
            future_timing, future_mark = components(model, t, altered)
            causal_error = max((timing[:, :6] - future_timing[:, :6]).abs().max().item(),
                               (mark[:, :6] - future_mark[:, :6]).abs().max().item())
            assert causal_error < 1e-10
            rows[name] = dict(decomposition_max_error=error, future_mark_past_score_error=causal_error,
                              finite_at_zero_gap=True)
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--tag', required=True); args = parser.parse_args()
    files = [Path(__file__).resolve(), Path(__file__).with_name('b10_components.py'),
             Path(__file__).with_name('b10_tpp.py'), ROOT / 'experiments/tpp/race_tpp_v5.py']
    result = dict(status='completed', battle='B10', tag=args.tag, checks=checks(),
                  source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    with (ROOT / 'experiments/results/market' / f'{args.tag}.json').open('x') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    print(json.dumps(result['checks']), flush=True)
