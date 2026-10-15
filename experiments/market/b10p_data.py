"""B10-P price task data: the B10 TRAIN/DEV windows with causal price/size/volatility inputs and tick-cell return labels.

Per event i of a window (all inputs use only trades at or before event i, from the full day stream):
  f0 log trade quantity (standardized with TRAIN moments)
  f1 signed log1p(|price change| in ticks) versus the previous trade
  f2..f4 log(1e-4 + realized variance in bp^2 of completed one-second returns over the past 60/300/1800 s)
  f5 log(1 + trades in the past 60 s)
  f6 signed aggressor-quantity imbalance over the past 10 s, in [-1, 1]
Labels for H in HORIZONS: the log-return in bp from the event's trade price to the last trade price at or before t_i + H,
as the exact tick cell [1e4 log((P_f - tick/2)/P_i), 1e4 log((P_f + tick/2)/P_i)]. Labels whose horizon passes the end of
the day are invalid. September is never opened.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
NPZ = ROOT / 'data/binance/npz'
CACHE = ROOT / 'data/binance/b10p'
TICK = 0.01
HORIZONS = (10.0, 60.0, 300.0)
NFEAT = 7
W = 1024


def day_arrays(day):
    """Full-day causal features and labels for one day; cached."""
    path = CACHE / f'{day}.npz'
    if path.exists():
        z = np.load(path); return {k: z[k] for k in z.files}
    z = np.load(NPZ / f'{day}.npz'); assert 'p' in z.files and 'q' in z.files, day
    t = z['t'].astype(np.float64) * 1e-6; mark = z['mark'].astype(np.int64); p = z['p'].astype(np.float64); q = z['q'].astype(np.float64)
    n = len(t); t0 = np.floor(t[0])
    sell = mark >= 3; signed = np.where(sell, -q, q)
    dp = np.diff(p, prepend=p[0]) / TICK
    # One-second last-trade price grid; second s covers (t0+s-1, t0+s].
    seconds = int(np.ceil(t[-1] - t0)) + 2
    edge = t0 + np.arange(seconds)
    last = np.searchsorted(t, edge, side='right') - 1
    grid = np.log(p[np.clip(last, 0, n - 1)]) * 1e4; grid[last < 0] = np.log(p[0]) * 1e4
    sq = np.concatenate([[0.], np.diff(grid) ** 2]); csq = np.cumsum(sq)
    s_now = np.floor(t - t0).astype(np.int64)                      # completed seconds strictly before t_i
    feats = np.zeros((n, NFEAT))
    feats[:, 0] = np.log(q)
    feats[:, 1] = np.sign(dp) * np.log1p(np.abs(dp))
    for j, L in enumerate((60, 300, 1800)):
        lo = np.clip(s_now - L, 0, None)
        feats[:, 2 + j] = np.log(1e-4 + csq[s_now] - csq[lo])
    idx = np.arange(n)
    feats[:, 5] = np.log1p(idx - np.searchsorted(t, t - 60.0, side='right') + 1)
    cs = np.concatenate([[0.], np.cumsum(signed)]); ca = np.concatenate([[0.], np.cumsum(q)])
    j10 = np.searchsorted(t, t - 10.0, side='right')
    feats[:, 6] = (cs[idx + 1] - cs[j10]) / (ca[idx + 1] - ca[j10])
    lo_lab = np.full((n, len(HORIZONS)), np.nan); hi_lab = np.full((n, len(HORIZONS)), np.nan)
    for h, H in enumerate(HORIZONS):
        valid = t + H <= t[-1]
        f = np.searchsorted(t, t + H, side='right') - 1
        pf = p[f]
        lo_lab[valid, h] = 1e4 * np.log((pf[valid] - TICK / 2) / p[valid])
        hi_lab[valid, h] = 1e4 * np.log((pf[valid] + TICK / 2) / p[valid])
    out = dict(t=t, mark=mark, feats=feats, lo=lo_lab, hi=hi_lab, price=p)
    CACHE.mkdir(parents=True, exist_ok=True); np.savez(path, **out)
    return out


def split_windows(days, stride):
    """Same windows as b10_depth_tpp_v1.windows(day, stride): consecutive 1024-event blocks, every stride-th."""
    rows = []
    for day in days:
        a = day_arrays(day); n = len(a['t']) // W
        for w in range(0, n, stride):
            s = slice(w * W, (w + 1) * W); tt = a['t'][s]
            rows.append(dict(day=str(day), window=w, t=tt - tt[0], mark=a['mark'][s], feats=a['feats'][s], lo=a['lo'][s], hi=a['hi'][s]))
    return rows


def standardize(train, *others):
    x = np.concatenate([r['feats'] for r in train]); mu = x.mean(0); sd = x.std(0) + 1e-9
    mu[6] = 0.; sd[6] = 1.                                          # imbalance already bounded
    for rows in (train, *others):
        for r in rows:
            r['feats'] = (r['feats'] - mu) / sd
    return dict(mean=mu.tolist(), std=sd.tolist())


def label_scales(train):
    """TRAIN standard deviation of each horizon's return (bp): initial head scale only."""
    out = []
    for h in range(len(HORIZONS)):
        v = np.concatenate([0.5 * (r['lo'][:, h] + r['hi'][:, h]) for r in train]); out.append(float(np.nanstd(v)))
    return out


def receipt(days):
    return {str(d): hashlib.sha256((CACHE / f'{d}.npz').read_bytes()).hexdigest() for d in days}


if __name__ == '__main__':
    import datetime as dt
    from b10_depth_tpp_v1 import TRAIN, VAL
    tr = split_windows(TRAIN, 20); dv = split_windows(VAL, 10)
    print(json.dumps(dict(train_windows=len(tr), dev_windows=len(dv), scales=label_scales(tr),
                          valid_dev=[float(np.mean([np.isfinite(r['lo'][128:, h]).mean() for r in dv])) for h in range(3)])))
