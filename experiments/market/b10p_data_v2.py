"""B10-P data v2: identical inputs and labels to b10p_data, but the causal look-back features (realized variance over
60/300/1800 s, trade count over 60 s, signed-flow imbalance over 10 s) also use the previous calendar day's final trades,
so the first 30 minutes of a day are not artificially truncated. Labels still end at the day's last trade."""
import datetime as dt
from pathlib import Path
import numpy as np
import b10p_data as v1
from b10p_data import NPZ, TICK, HORIZONS, NFEAT, W, standardize, label_scales

CACHE = v1.ROOT/'data/binance/b10p_v2'
LOOKBACK = 1800.0


def day_arrays(day):
    path = CACHE/f'{day}.npz'
    if path.exists():
        z = np.load(path); return {k: z[k] for k in z.files}
    z = np.load(NPZ/f'{day}.npz'); t = z['t'].astype(np.float64)*1e-6; mark = z['mark'].astype(np.int64); p = z['p'].astype(np.float64); q = z['q'].astype(np.float64)
    previous = NPZ/f'{dt.date.fromisoformat(str(day)) - dt.timedelta(days=1)}.npz'
    if previous.exists() and 'p' in np.load(previous).files:
        y = np.load(previous); ty = y['t'].astype(np.float64)*1e-6; keep = ty > t[0] - LOOKBACK - 2
        T = np.concatenate([ty[keep], t]); P = np.concatenate([y['p'][keep].astype(np.float64), p]); Q = np.concatenate([y['q'][keep].astype(np.float64), q]); M = np.concatenate([y['mark'][keep].astype(np.int64), mark]); offset = int(keep.sum())
    else:
        T, P, Q, M, offset = t, p, q, mark, 0
    n = len(T); t0 = np.floor(T[0]); signed = np.where(M >= 3, -Q, Q)
    seconds = int(np.ceil(T[-1] - t0)) + 2; edge = t0 + np.arange(seconds)
    last = np.searchsorted(T, edge, side='right') - 1
    grid = np.log(P[np.clip(last, 0, n - 1)])*1e4; grid[last < 0] = np.log(P[0])*1e4
    csq = np.cumsum(np.concatenate([[0.], np.diff(grid)**2])); s_now = np.floor(T - t0).astype(np.int64)
    feats = np.zeros((n, NFEAT)); dp = np.diff(P, prepend=P[0])/TICK
    feats[:, 0] = np.log(Q); feats[:, 1] = np.sign(dp)*np.log1p(np.abs(dp))
    for j, L in enumerate((60, 300, 1800)):
        feats[:, 2 + j] = np.log(1e-4 + csq[s_now] - csq[np.clip(s_now - L, 0, None)])
    idx = np.arange(n); feats[:, 5] = np.log1p(idx - np.searchsorted(T, T - 60.0, side='right') + 1)
    cs = np.concatenate([[0.], np.cumsum(signed)]); ca = np.concatenate([[0.], np.cumsum(Q)]); j10 = np.searchsorted(T, T - 10.0, side='right')
    feats[:, 6] = (cs[idx + 1] - cs[j10])/(ca[idx + 1] - ca[j10])
    feats = feats[offset:]
    lo = np.full((len(t), len(HORIZONS)), np.nan); hi = np.full_like(lo, np.nan)
    for h, H in enumerate(HORIZONS):
        valid = t + H <= t[-1]; f = np.searchsorted(t, t + H, side='right') - 1; pf = p[f]
        lo[valid, h] = 1e4*np.log((pf[valid] - TICK/2)/p[valid]); hi[valid, h] = 1e4*np.log((pf[valid] + TICK/2)/p[valid])
    out = dict(t=t, mark=mark, feats=feats, lo=lo, hi=hi, price=p, lookback_trades=np.array(offset))
    CACHE.mkdir(parents=True, exist_ok=True); np.savez(path, **out)
    return out
