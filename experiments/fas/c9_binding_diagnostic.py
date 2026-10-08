#!/usr/bin/env python3
"""B3 diagnostic (evaluation only; opens the identity sidecar, as the protocol allows for diagnostics): does C9's
learned predecessor attention find each event's true in-line predecessor?

For validation-clean runs, for every process event j with a same-(line, item) predecessor inside the attention window,
report the attention mass (per head, and the best head) on (a) that true item predecessor, (b) any event of the same
line, (c) the immediately preceding merged event; with the uniform-attention reference 1/W.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/fas')); sys.path.insert(0, str(ROOT / 'experiments/tpp'))
import race_tpp_fas_v2 as R  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--result', required=True)
    ap.add_argument('--runs', type=int, default=200)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    torch.set_default_dtype(torch.float64); torch.set_num_threads(1)
    res = json.loads((ROOT / a.result).read_text()); args = res['args']
    d = ROOT / 'experiments/data/fas' / args['data']
    z = np.load(d / 'val_clean.npz'); o, ids_all, t_all = z['offsets'], z['ids'], z['times_ms']
    idn = np.load(d / 'identity.npz'); line_all, item_all = idn['val_clean_line'], idn['val_clean_item']
    train, _ = R.load(d / 'train_clean.npz', args['train_max_events']); train = train[:args['fit_runs']]
    gaps = np.concatenate([np.diff(ts) for _, ts in train[:500]]); pos = gaps[gaps > 0]
    scale = float(np.median(pos)); qs = np.log(np.quantile(pos, np.linspace(0.1, 0.9, args['n_lognormal']))).tolist()
    edges = R.cluster_windows(pos, args['n_window'], args['seed']) if args['n_window'] else None
    margs = (46, args['d'], args['modes'], args['layers'], args['n_exp'], args['n_lognormal'], args['dv'], args['dropout'],
             scale, qs, args['cell_s'], args['n_window'], edges, 0)
    model = R.KeyedRaceFAS(*margs, dk=args['dk'], local=args['keyed'] == 2, pred_window=args['pred_window'],
                           keyed=bool(args['keyed']))
    model.load_state_dict(torch.load(ROOT / res['checkpoint'])); model.eval()
    P = model.pred; W = P.W
    acc = dict(true_item=[], same_line=[], prev_merged=[], in_window=0, events=0)
    with torch.no_grad():
        for r in range(a.runs):
            sl = slice(o[r], o[r + 1]); ids = ids_all[sl].astype(np.int64); keep = ids != 0
            ids = ids[keep][:args['max_events']]; t = (t_all[sl][keep] / 1000.)[:args['max_events']]
            line = line_all[sl][keep][:len(ids)]; item = item_all[sl][keep][:len(ids)]
            L = len(ids); m = torch.from_numpy(ids)[None]; tt = torch.from_numpy(t)[None]
            pad_m = torch.cat([m.new_zeros(1, W), m], 1); pad_t = torch.cat([tt[:, :1].expand(1, W), tt], 1)
            idx = torch.arange(L)[:, None] + torch.arange(W)[None, :]
            mi = pad_m[:, idx]; ti = pad_t[:, idx]
            valid = (torch.arange(L)[:, None] - W + torch.arange(W)[None, :]) >= 0
            lg = torch.log1p((tt[:, :, None] - ti).clamp_min(0) / scale)
            gf = torch.stack([lg, lg ** 2, (lg == 0).to(lg.dtype)], -1)
            q = P.q(m).view(1, L, 1, P.h, P.dq); k = P.k(mi).view(1, L, W, P.h, P.dq)
            sc = (q * k).sum(-1) / math.sqrt(P.dq) + P.gap_w(gf)
            att = torch.softmax(sc.masked_fill(~valid[None, :, :, None], -1e9), 2)[0].numpy()   # [L, W, h]
            for j in range(1, L):
                prev = [i for i in range(max(0, j - W), j) if line[i] == line[j] and item[i] == item[j]]
                acc['events'] += 1
                if not prev:
                    continue
                acc['in_window'] += 1
                ti_ = prev[-1]; w_true = ti_ - (j - W)
                same = [i - (j - W) for i in range(max(0, j - W), j) if line[i] == line[j]]
                acc['true_item'].append(att[j, w_true, :].tolist())
                acc['same_line'].append(att[j, same, :].sum(0).tolist())
                acc['prev_merged'].append(att[j, W - 1, :].tolist())
    out = dict(result=a.result, runs=a.runs, window=W, uniform_reference=1 / W,
               fraction_with_item_predecessor_in_window=acc['in_window'] / max(acc['events'], 1))
    for key in ('true_item', 'same_line', 'prev_merged'):
        x = np.array(acc[key]); out[key] = dict(mean_per_head=x.mean(0).round(4).tolist(), best_head=float(x.mean(0).max()))
    p = ROOT / a.out; p.write_text(json.dumps(out, indent=1) + '\n'); print(json.dumps(out))


if __name__ == '__main__':
    main()
