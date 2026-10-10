#!/usr/bin/env python3
"""tgbl-review causal driver v6 = v5 + sampled-softmax log-Q correction during training.

Failure (v5 arms, 10 Oct): A (no identities + popularity residual) 0.261 and B (+ hard negatives from recent destination
events) 0.112 on full VALIDATION, both below 30-day popularity alone (0.341). Diagnosis: the training loss is a softmax over the
positive and competitors drawn from a non-uniform proposal q; without correction it learns score - log q, i.e. against the
proposal. With popularity-proportional (hard) negatives that means against popularity, matching B's collapse.
  --logq   training logits become score_j - log q(j), q(j) = probability that one competitor draw yields j:
           q(j) = 1/2 * q_hist(j) + 1/2 * (1/n),  q_hist = ring frequency (hard negatives) or 1/|seen| for seen j (v3 mode).
           Applied to every candidate (positive included), training only; evaluation and the official Evaluator are unchanged.
Approximation: log q is computed once per 200-query training batch, after the causal state has advanced through it (the
seen set / ring may be slightly ahead of early queries in the batch). Without --logq v6 is v5. No TEST flag.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import race_link_review_v5 as v5  # noqa: E402
import race_link_review as base  # noqa: E402
import race_link_review_v2 as v2  # noqa: E402

LOGQ = dict(on=False, stash=None)
_v5_add = base.State.add
_v2_causal_batch = v2.causal_batch


def add(self, s, c, t):
    """v5 add (ring of recent destination events) plus exact ring counts for q_hist."""
    if v5.HARD['on'] and LOGQ['on']:
        R = v5.HARD['R']
        cnt = getattr(self, '_ring_cnt', None)
        if cnt is None:
            self._ring_cnt = cnt = np.zeros(self.n, np.int64)
        start = getattr(self, '_ring_n', 0)
        ring = getattr(self, '_ring', None)
        evicted = []
        for k in range(len(c)):
            pos = start + k
            if pos >= R and ring is not None:
                evicted.append(int(ring[pos % R]))
        _v5_add(self, s, c, t)
        for p in np.asarray(c, np.int64):
            cnt[p] += 1
        for p in evicted:
            cnt[p] -= 1
    else:
        _v5_add(self, s, c, t)


def log_q(st, cand):
    n = st.n
    if v5.HARD['on']:
        m = max(min(getattr(st, '_ring_n', 0), v5.HARD['R']), 1)
        cnt = getattr(st, '_ring_cnt', np.zeros(n, np.int64))
        q_hist = cnt[cand] / m
    else:
        L = max(len(st.seen_list), 1)
        q_hist = st.seen_dst[cand].astype(float) / L
    return np.log(0.5 * q_hist + 0.5 / n)


def causal_batch(st, s, c, t, candidates=None, neg=20, rng=None):
    cand, feat = _v2_causal_batch(st, s, c, t, candidates=candidates, neg=neg, rng=rng)
    LOGQ['stash'] = log_q(st, cand) if (LOGQ['on'] and candidates is None) else None
    return cand, feat


v2.causal_batch = causal_batch
base.State.add = add
_ModelV4 = base.Model


class Model(_ModelV4):
    def forward(self, s, cand, feat):
        x = super().forward(s, cand, feat)
        st = LOGQ['stash']
        if self.training and st is not None and st.shape == tuple(cand.shape):
            x = x - torch.from_numpy(st).to(x.dtype)
            LOGQ['stash'] = None
        return x


base.Model = Model


def main():
    if '--logq' in sys.argv:
        sys.argv.remove('--logq'); LOGQ['on'] = True
    tag = sys.argv[sys.argv.index('--tag') + 1]
    v5.main()
    out = v2.ROOT / 'experiments/results/tgb' / f'{tag}.json'
    res = json.loads(out.read_text())
    res['driver'] = 'race_link_review_v6.py (v5 + sampled-softmax log-Q correction in training)'
    res['logq'] = LOGQ['on']
    res.setdefault('source_sha256', {})[str(Path(__file__).resolve().relative_to(v2.ROOT))] = \
        hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    out.write_text(json.dumps(res, indent=1) + '\n')


if __name__ == '__main__':
    main()
