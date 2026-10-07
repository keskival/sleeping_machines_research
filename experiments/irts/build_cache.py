#!/usr/bin/env python3
"""B2: convert Raindrop releases (via the allow-list loader) into compact arrays; no model fitting.

Per record: observation times (hours), channel values, observation mask (Raindrop convention: a channel is observed at
a time step when its value is non-zero), static covariates, label; plus the five official split index triples.
Output: data/raindrop/cache/{P12,P19}.npz
"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from safe_load import load_npy  # noqa: E402

SPEC = {
    'P12': dict(zip='data/raindrop/P12/P12data.zip', records='P12data/processed_data/PTdict_list.npy',
                outcomes='P12data/processed_data/arr_outcomes.npy', label_col=-1,
                splits=[f'P12data/splits/phy12_split{k}.npy' for k in range(1, 6)], time_scale=1 / 60),
    'P19': dict(zip='data/raindrop/P19/P19data.zip', records='P19data/processed_data/PT_dict_list_6.npy',
                outcomes='P19data/processed_data/arr_outcomes_6.npy', label_col=0,
                splits=[f'P19data/splits/phy19_split{k}_new.npy' for k in range(1, 6)], time_scale=1.0),
}


def main_pam():
    z = ROOT / 'data/raindrop/PAM/PAMdata.zip'
    arr = load_npy(z, 'PAMAP2data/processed_data/PTdict_list.npy').astype(np.float32)        # [N, 600, 17]
    y = load_npy(z, 'PAMAP2data/processed_data/arr_outcomes.npy')[:, 0].astype(np.int64)
    N, T, C = arr.shape
    times = np.tile(np.arange(T, dtype=np.float32), (N, 1))                                # time = step index
    lens = np.full(N, T, np.int64); static = np.zeros((N, 1), np.float32)
    splits = []
    for k in range(1, 6):
        tr, va, te = load_npy(z, f'PAMAP2data/splits/PAMAP2_split_{k}.npy')
        splits.append((np.asarray(tr, np.int64), np.asarray(va, np.int64), np.asarray(te, np.int64)))
    out = ROOT / 'data/raindrop/cache'; out.mkdir(exist_ok=True)
    np.savez_compressed(out / 'PAM.npz', times=times, vals=arr, mask=arr != 0, lens=lens, static=static, y=y,
                        **{f'split{k}_{part}': idx for k, sp in enumerate(splits) for part, idx in zip(('train', 'val', 'test'), sp)})
    print('PAM records', N, 'T', T, 'channels', C, 'classes', np.bincount(y), 'obs frac', round(float((arr != 0).mean()), 3),
          'split sizes', [tuple(len(x) for x in sp) for sp in splits])


def main(name):
    if name == 'PAM':
        return main_pam()
    s = SPEC[name]
    recs = load_npy(ROOT / s['zip'], s['records'])
    y = load_npy(ROOT / s['zip'], s['outcomes'])[:, s['label_col']].astype(np.int64)
    T = max(int(r['length']) for r in recs); C = recs[0]['arr'].shape[1]
    N = len(recs)
    times = np.zeros((N, T), np.float32); vals = np.zeros((N, T, C), np.float32)
    lens = np.zeros(N, np.int64); static = np.stack([np.asarray(r['extended_static'], np.float32) for r in recs])
    for i, r in enumerate(recs):
        L = int(r['length']); lens[i] = L
        times[i, :L] = np.asarray(r['time'], np.float32)[:L, 0] * s['time_scale']
        vals[i, :L] = np.asarray(r['arr'], np.float32)[:L]
    mask = vals != 0
    splits = []
    for p in s['splits']:
        tr, va, te = load_npy(ROOT / s['zip'], p)
        splits.append((np.asarray(tr, np.int64), np.asarray(va, np.int64), np.asarray(te, np.int64)))
    out = ROOT / 'data/raindrop/cache'; out.mkdir(exist_ok=True)
    np.savez_compressed(out / f'{name}.npz', times=times, vals=vals, mask=mask, lens=lens, static=static, y=y,
                        **{f'split{k}_{part}': idx for k, sp in enumerate(splits) for part, idx in zip(('train', 'val', 'test'), sp)})
    print(name, 'records', N, 'T', T, 'channels', C, 'static', static.shape[1], 'positive rate', y.mean().round(4),
          'obs per record', mask.sum((1, 2)).mean().round(1), 'max time', times.max().round(2),
          'split sizes', [tuple(len(x) for x in sp) for sp in splits])


if __name__ == '__main__':
    main(sys.argv[1])
