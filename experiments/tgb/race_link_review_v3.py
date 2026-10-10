#!/usr/bin/env python3
"""tgbl-review causal driver v3 = race_link_review_v2.py unchanged, with one engineering repair.

Failure addressed (measured 10 Oct 2026 with py-spy on the v2 full-development epoch): State.sample_negatives rebuilt
np.asarray(self.seen_list) for every training query, so one epoch was quadratic in TRAIN events (2.17M of 3.41M after
3 h 13 min; the 50K-event pilot could not show it). v3 keeps the seen-destination array cached and extends it
amortized. Contents and order are identical, so every random draw, feature, loss and score is bitwise identical to v2;
`--expect-val-mrr/--expect-train-loss` assert this against a completed v2 run. Model, protocol, causal state, sampled
competitors and losing-candidate credit are unchanged. No TEST flag (as v2).
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import race_link_review as base  # noqa: E402


def sample_negatives(self, k, rng, Q):
    L = len(self.seen_list)
    arr = getattr(self, '_seen_arr', None); m = getattr(self, '_seen_n', 0)
    if arr is None or L > len(arr):
        new = np.empty(max(1024, 2 * L), np.int64)
        if arr is not None:
            new[:m] = arr[:m]
        arr = new
    if L > m:
        arr[m:L] = self.seen_list[m:L]
    self._seen_arr, self._seen_n = arr, L
    hist = arr[:L]
    kh = k // 2 if L else 0
    h = hist[rng.integers(0, L, (Q, kh))] if kh else np.zeros((Q, 0), np.int64)
    r = rng.integers(0, self.n, (Q, k - kh))
    return np.concatenate([h, r], 1)


base.State.sample_negatives = sample_negatives
import race_link_review_v2 as v2  # noqa: E402


def main():
    expect = {}
    for flag in ('--expect-val-mrr', '--expect-train-loss'):
        if flag in sys.argv:
            i = sys.argv.index(flag); expect[flag] = float(sys.argv[i + 1]); del sys.argv[i:i + 2]
    tag = sys.argv[sys.argv.index('--tag') + 1]
    v2.main()
    out = v2.ROOT / 'experiments/results/tgb' / f'{tag}.json'
    res = json.loads(out.read_text())
    res['driver'] = 'race_link_review_v3.py (v2 + amortized seen-destination array; bitwise-identical draws)'
    res.setdefault('source_sha256', {})[str(Path(__file__).resolve().relative_to(v2.ROOT))] = \
        hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    checks = {}
    if '--expect-val-mrr' in expect:
        checks['val_mrr'] = dict(expected=expect['--expect-val-mrr'], got=res['val_mrr'],
                                 equal=res['val_mrr'] == expect['--expect-val-mrr'])
    if '--expect-train-loss' in expect:
        got = res['history'][0]['train_loss']
        checks['train_loss_epoch0'] = dict(expected=expect['--expect-train-loss'], got=got, equal=got == expect['--expect-train-loss'])
    if checks:
        res['v2_equivalence'] = checks
        if not all(c['equal'] for c in checks.values()):
            res['status'] = 'failed'
    out.write_text(json.dumps(res, indent=1) + '\n')
    print('V3', json.dumps(checks), res['status'], flush=True)
    if res['status'] != 'completed':
        sys.exit(1)


if __name__ == '__main__':
    main()
