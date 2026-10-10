#!/usr/bin/env python3
"""tgbl-review causal driver v5 = v4 (identity / pop-residual switches over v3) + hard historical negatives.

Failure (ablation of the v3 checkpoint, 10 Oct): the trained network ranks below 30-day popularity alone (0.277 vs 0.341)
and degrades it when added (0.313). Training competitors were half uniform over all nodes and half uniform over ever-seen
destinations: almost all are separable by seen/recency flags, so ranking among popular candidates, which the official
negatives require, was never learned.
  --hard-negatives [R]   the historical half of each query's competitors is drawn uniformly from the last R destination
                         EVENTS (default R = 200000), i.e. in proportion to recent popularity, instead of uniformly over
                         ever-seen destinations. The random half and the duplicate/positive exclusion are unchanged.
Without the flag v5 is v4 (and with v4 defaults, bitwise v3). No TEST flag.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import race_link_review_v4 as v4  # noqa: E402  (installs v3 sampler and the switchable Model)
import race_link_review as base  # noqa: E402

HARD = dict(on=False, R=200000)
_orig_add = base.State.add
_v3_sample = base.State.sample_negatives


def add(self, s, c, t):
    _orig_add(self, s, c, t)
    if HARD['on']:
        ring = getattr(self, '_ring', None)
        if ring is None:
            self._ring = ring = np.zeros(HARD['R'], np.int64); self._ring_n = 0
        for p in np.asarray(c, np.int64):
            ring[self._ring_n % HARD['R']] = p; self._ring_n += 1


def sample_negatives(self, k, rng, Q):
    if not HARD['on']:
        return _v3_sample(self, k, rng, Q)
    m = min(getattr(self, '_ring_n', 0), HARD['R'])
    kh = k // 2 if m else 0
    h = self._ring[rng.integers(0, m, (Q, kh))] if kh else np.zeros((Q, 0), np.int64)
    r = rng.integers(0, self.n, (Q, k - kh))
    return np.concatenate([h, r], 1)


base.State.add = add
base.State.sample_negatives = sample_negatives


def main():
    if '--hard-negatives' in sys.argv:
        i = sys.argv.index('--hard-negatives'); HARD['on'] = True
        if i + 1 < len(sys.argv) and sys.argv[i + 1].isdigit():
            HARD['R'] = int(sys.argv[i + 1]); del sys.argv[i:i + 2]
        else:
            del sys.argv[i]
    tag = sys.argv[sys.argv.index('--tag') + 1]
    v4.main()
    out = v4.v3.v2.ROOT / 'experiments/results/tgb' / f'{tag}.json'
    res = json.loads(out.read_text())
    res['driver'] = 'race_link_review_v5.py (v4 + hard historical negatives from recent destination events)'
    res['hard_negatives'] = dict(HARD)
    res.setdefault('source_sha256', {})[str(Path(__file__).resolve().relative_to(v4.v3.v2.ROOT))] = \
        hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    out.write_text(json.dumps(res, indent=1) + '\n')


if __name__ == '__main__':
    main()
