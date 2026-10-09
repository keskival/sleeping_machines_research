#!/usr/bin/env python3
"""Theory note 158 §455 Prediction 2: log-likelihood improves with the number of delayed clocks until the gap distribution's
components are covered, then saturates (Theorem 1 at finite size).

Data: one renewal stream per sequence with K = 4 marks; gaps drawn from a mixture of G log-normal components with means
spread log-uniformly over [0.1, 30] and log-sd 0.15 (well-separated components); the mark is tied to the component (mark =
component mod 4), so mark and timing interact. G ∈ {2, 6}. Model: race_tpp_v5 with 1 exponential clock and n_lognormal ∈
{1, 2, 4, 8, 16} delayed clocks (same memory). Score: TEST log-likelihood per event; the prediction is saturation near
n_lognormal ≈ G.
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from race_tpp_v5 import RaceTPP, batches, evaluate  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def make(G, n, length, rng):
    means = np.exp(np.linspace(math.log(0.1), math.log(30.0), G)); w = np.ones(G) / G
    seqs = []
    for _ in range(n):
        comp = rng.choice(G, length, p=w); gaps = rng.lognormal(np.log(means[comp]), 0.15)
        t = np.concatenate([[0.0], np.cumsum(gaps[1:])]); seqs.append((t, (comp % 4).astype(np.int64)))
    return seqs


def fit(train, dev, test, n_ln, seed, epochs, max_wall):
    torch.manual_seed(seed); rng = random.Random(seed)
    gaps = np.concatenate([np.diff(t) for t, _ in train]); pos = gaps[gaps > 0]; scale = float(np.median(pos))
    qs = np.log(np.quantile(pos, np.linspace(0.1, 0.9, n_ln))).tolist()
    model = RaceTPP(4, 32, 16, 2, 1, n_ln, 4, 0.1, scale, qs)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3); best, state, t0 = -math.inf, None, time.time()
    for _ in range(epochs):
        model.train()
        for t, m, mask in batches(train, 32, True, rng):
            tl, ml, n, _ = model.loglik(t, m, mask); loss = -(tl + ml) / n
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        dv = evaluate(model, dev, 64, rng, predict=False)['ll']
        if dv > best:
            best, state = dv, {k: v.clone() for k, v in model.state_dict().items()}
        if time.time() - t0 > max_wall:
            break
    model.load_state_dict(state)
    return evaluate(model, test, 64, rng, predict=False)['ll']


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=25); ap.add_argument('--max-wall-s', type=float, default=400)
    a = ap.parse_args(); torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
    rows = []
    for G in (2, 6):
        rng = np.random.default_rng(10 + G + a.seed)
        tr, dv, te = make(G, 500, 40, rng), make(G, 120, 40, rng), make(G, 120, 40, rng)
        for n_ln in (1, 2, 4, 8, 16):
            ll = fit(tr, dv, te, n_ln, a.seed, a.epochs, a.max_wall_s)
            rows.append(dict(G=G, n_lognormal=n_ln, test_ll=ll)); print(json.dumps(rows[-1]), flush=True)
    (ROOT / f'experiments/results/tpp/{a.tag}.json').write_text(json.dumps(dict(status='completed', tag=a.tag,
        theory='note 158 sect. 455 prediction 2', args=vars(a), rows=rows), indent=1) + '\n')


if __name__ == '__main__':
    main()
