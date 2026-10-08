#!/usr/bin/env python3
"""Generative mode for the race-of-clocks event model (race_tpp_v5): exact sampling by racing the clocks.

After each event every clock draws a firing time: an exponential clock fires at Exp(rate · w); a defective delayed clock
fires with probability π at LogNormal(μ, σ) and otherwise never. The earliest clock gives the next event's time, and its
own mark law gives the type. This is exactly the competing-risks process whose likelihood the model is trained on, so
sampling needs no thinning, rejection or numerical inversion. Each generated event is fed back into the memory.

Evaluation against held-out TEST sequences (conditioning on each test sequence's first event, generating the same number
of events): mark frequencies and mark-to-mark transitions (total variation distances), log-gap quantiles and a two-sample KS statistic on gaps, and the
same statistics for a shuffled-marks / exponential-gaps reference to calibrate how far "unrealistic" sits.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from race_tpp_v5 import RaceTPP, load_split  # noqa: E402


@torch.no_grad()
def sample_sequence(model, t0, m0, n_events, gen):
    t = [float(t0)]; m = [int(m0)]
    for _ in range(n_events - 1):
        T = torch.tensor([t], dtype=torch.float64); Mk = torch.tensor([m]); mask = torch.ones_like(Mk, dtype=torch.bool)
        h, slots = model.encode(T, Mk, mask)
        log_rate, mu, sigma, log_w, log_pk = (x[0, -1] for x in model.clocks(h[:, -1:], slots[:, -1:]))
        ne = log_rate.shape[0]
        rate = (log_rate + log_w[:ne]).exp()                                     # exponential clocks: rate · weight
        taus = [float(torch.empty(1, dtype=torch.float64).exponential_(1.0, generator=gen)) / float(r) for r in rate]
        pi = log_w[ne:].exp()
        for k in range(len(mu)):                                                 # defective delayed clocks
            fires = float(torch.rand(1, generator=gen, dtype=torch.float64)) < float(pi[k])
            z = float(torch.randn(1, generator=gen, dtype=torch.float64))
            taus.append(math.exp(float(mu[k]) + float(sigma[k]) * z) if fires else math.inf)
        win = int(np.argmin(taus))
        mark = int(torch.multinomial(log_pk[win].exp(), 1, generator=gen))
        t.append(t[-1] + taus[win]); m.append(mark)
    return np.array(t), np.array(m)


def summarize(seqs, K):
    marks = np.concatenate([m for _, m in seqs]); gaps = np.concatenate([np.diff(t) for t, _ in seqs])
    freq = np.bincount(marks, minlength=K) / len(marks)
    lg = np.log(np.clip(gaps, 1e-6, None))
    trans = np.zeros((K, K))
    for _, m in seqs:
        np.add.at(trans, (m[:-1], m[1:]), 1)
    return freq, gaps, np.quantile(lg, [0.1, 0.25, 0.5, 0.75, 0.9]), trans / trans.sum()


def ks(a, b):
    a, b = np.sort(a), np.sort(b); x = np.concatenate([a, b])
    return float(np.max(np.abs(np.searchsorted(a, x, 'right') / len(a) - np.searchsorted(b, x, 'right') / len(b))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--result', required=True, help='a race_tpp_v5 result JSON with its checkpoint')
    ap.add_argument('--sequences', type=int, default=200)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    torch.set_default_dtype(torch.float64); torch.set_num_threads(1)
    res = json.loads((ROOT / a.result).read_text()); args = res['args']
    train = load_split(args['dataset'], 'train'); test = load_split(args['dataset'], 'test')[:a.sequences]
    K = res['K']; gaps = np.concatenate([np.diff(t) for t, _ in train]); pos = gaps[gaps > 0]
    qs = np.log(np.quantile(pos, np.linspace(0.1, 0.9, args['n_lognormal']))).tolist()
    model = RaceTPP(K, args['d'], args['modes'], args['layers'], args['n_exp'], args['n_lognormal'], args['dv'],
                    args['dropout'], res['scale'], qs, args.get('floor_cell', 0.0))
    model.load_state_dict(torch.load(ROOT / res['checkpoint'])); model.eval()
    gen = torch.Generator().manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    generated = [sample_sequence(model, t[0], m[0], len(m), gen) for t, m in test]
    # reference: marks drawn from the TRAIN marginal, exponential gaps with the TRAIN mean gap
    tm = np.concatenate([m for _, m in train]); pm = np.bincount(tm, minlength=K) / len(tm); mg = gaps.mean()
    naive = [(np.concatenate([[t[0]], t[0] + np.cumsum(rng.exponential(mg, len(m) - 1))]), rng.choice(K, len(m), p=pm))
             for t, m in test]
    out = dict(result=a.result, dataset=args['dataset'], sequences=len(test), events=int(sum(len(m) for _, m in test)))
    f_real, g_real, q_real, tr_real = summarize(test, K)
    for name, seqs in (('generated', generated), ('naive_reference', naive)):
        f, g, q, tr = summarize(seqs, K)
        out[name] = dict(mark_tv_distance=float(0.5 * np.abs(f - f_real).sum()), gap_ks=ks(g, g_real),
                         transition_tv_distance=float(0.5 * np.abs(tr - tr_real).sum()),
                         log_gap_quantiles=q.round(3).tolist())
    out['real'] = dict(log_gap_quantiles=q_real.round(3).tolist(), mark_freq=f_real.round(4).tolist())
    out['example'] = dict(real=[(round(float(x), 3), int(y)) for x, y in zip(*test[0])][:12],
                          generated=[(round(float(x), 3), int(y)) for x, y in zip(*generated[0])][:12])
    p = ROOT / a.out; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps({k: out[k] for k in ('generated', 'naive_reference', 'real')}))


if __name__ == '__main__':
    main()
