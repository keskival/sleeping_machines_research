#!/usr/bin/env python3
"""Theory note 158 §455 Prediction 5: the race head's advantage grows with the number of superposed processes.

Data: K independent renewal processes merged into one log. Process j emits only mark j; its inter-event durations are
LogNormal(log d_j, 0.25) with d_j spread log-uniformly over [0.5, 8]; each sequence covers a fixed horizon. The exact
conditional intensity depends on the time since *each* process's own last event, so a single clock restarted at every
merged event cannot represent it, while a race of several delayed clocks plus the temporal memory can.

Models (race_tpp_v5, same memory and size): FULL = 2 exponential + 8 defective delayed clocks; SINGLE = 1 exponential
clock (a single-intensity head with the same state). Score: TEST log-likelihood per event (time + mark), and the advantage
FULL − SINGLE as a function of K.
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from race_tpp_v5 import RaceTPP, batches, evaluate  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def make(K, n, horizon, rng):
    d = np.exp(np.linspace(math.log(0.5), math.log(8.0), K)) if K > 1 else np.array([2.0])
    seqs = []
    for _ in range(n):
        ev = []
        for j in range(K):
            t = rng.uniform(0, d[j])
            while t < horizon:
                ev.append((t, j)); t += rng.lognormal(math.log(d[j]), 0.25)
        ev.sort(); ts = np.array([e[0] for e in ev]); ms = np.array([e[1] for e in ev], np.int64)
        seqs.append((ts - ts[0], ms))
    return seqs


def fit(train, dev, test, K, n_exp, n_ln, seed, epochs, max_wall):
    torch.manual_seed(seed); random.seed(seed); rng = random.Random(seed)
    gaps = np.concatenate([np.diff(t) for t, _ in train]); pos = gaps[gaps > 0]; scale = float(np.median(pos))
    qs = np.log(np.quantile(pos, np.linspace(0.1, 0.9, max(n_ln, 1)))).tolist()[:n_ln]
    model = RaceTPP(max(K, 2), 32, 16, 2, n_exp, n_ln, 4, 0.1, scale, qs)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    best, state, t0 = -math.inf, None, time.time()
    for ep in range(epochs):
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
    return evaluate(model, test, 64, rng, predict=False), sum(p.numel() for p in model.parameters())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--ks', default='1,2,4,8')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=25)
    ap.add_argument('--max-wall-s', type=float, default=600)
    a = ap.parse_args()
    torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
    rows = []
    for K in [int(k) for k in a.ks.split(',')]:
        rng = np.random.default_rng(100 + K + a.seed); horizon = 40.0
        train, dev, test = make(K, 600, horizon, rng), make(K, 150, horizon, rng), make(K, 150, horizon, rng)
        full, pf = fit(train, dev, test, K, 2, 8, a.seed, a.epochs, a.max_wall_s)
        single, ps = fit(train, dev, test, K, 1, 0, a.seed, a.epochs, a.max_wall_s)
        rows.append(dict(K=K, events_per_seq=float(np.mean([len(m) for _, m in test])), full_ll=full['ll'],
                         single_ll=single['ll'], advantage=full['ll'] - single['ll'], full_params=pf, single_params=ps))
        print(json.dumps(rows[-1]), flush=True)
    out = ROOT / f'experiments/results/tpp/{a.tag}.json'
    out.write_text(json.dumps(dict(status='completed', tag=a.tag, theory='note 158 sect. 455 prediction 5', args=vars(a),
                                   rows=rows), indent=1) + '\n')
    print('RESULT', json.dumps([(r['K'], round(r['advantage'], 4)) for r in rows]))


if __name__ == '__main__':
    main()
