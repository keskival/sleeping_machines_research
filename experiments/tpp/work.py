#!/usr/bin/env python3
"""B1 resource accounting: parameters and per-event inference work, ours versus S2P2 at its published configurations.

Ours: parameters counted from the instantiated model; inference multiply-accumulates (MACs) per event counted from the
architecture (encoder update after an event plus the clock/mark head that defines the next-event distribution).
S2P2: parameters and MACs counted analytically from the released LLH layer definitions (UCIDataLab fork, commit
c3933240; complex weights count as two reals, a complex MAC as four real MACs) at the per-dataset architectures of the
S2P2 paper, Table 6a. One event's update, not the Monte Carlo compensator, is counted for both.
"""
import argparse, importlib, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
S2P2_CONFIG = {'amazon': (64, 128, 2), 'retweet': (128, 128, 2), 'taxi': (128, 16, 4), 'taobao': (32, 16, 4),
               'stackoverflow': (32, 32, 3)}           # (H, P, layers), S2P2 Table 6a
K_OF = {'amazon': 16, 'retweet': 3, 'taxi': 10, 'taobao': 17, 'stackoverflow': 22}


def s2p2(ds):
    H, P, L = S2P2_CONFIG[ds]; K = K_OF[ds]
    layer_params = 2 * P + 2 * P + P + 3 * (2 * P * H) + H * H + 2 * H + (2 * H * H + 2 * H) + 2 * H
    params = L * layer_params + (K + 1) * H + (H * K + K)
    # per event and layer: decay/rotate P complex states (4 real MACs each), B u and E u (P x H complex-by-real:
    # 2 real MACs per entry each), C x (H x P complex: 4 real MACs per entry, real part kept: 2), D u (H^2),
    # full GLU (2 H^2), layer norm (~5H)
    layer_macs = 4 * P + 2 * (2 * P * H) + 2 * H * P + H * H + 2 * H * H + 5 * H
    macs = L * layer_macs + H * K
    return dict(params=params, macs_per_event=macs, config=dict(H=H, P=P, layers=L))


def ours(result):
    r = json.loads((ROOT / result).read_text()); a = r['args']
    mod = importlib.import_module(Path(next(iter(r['source_sha256']))).stem)
    kw = {'floor_cell': a['floor_cell']} if 'floor_cell' in a else {}
    m = mod.RaceTPP(r['K'], a['d'], a['modes'], a['layers'], a['n_exp'], a['n_lognormal'], a['dv'], 0.0, r['scale'],
                    [0.0] * a['n_lognormal'], **kw)
    params = sum(p.numel() for p in m.parameters())
    d, n, L, K, dv = a['d'], a['modes'], a['layers'], r['K'], a['dv']
    M, ne, nl = a['n_exp'] + a['n_lognormal'], a['n_exp'], a['n_lognormal']
    layer = 2 * n * d + n * d + 6 * n + 2 * n * d + 4 * d * d + 10 * d
    head = (d + K * dv) * (ne + 3 * nl + M) + d * M * K + M * K * dv
    macs = 2 * d + L * layer + d * dv + 2 * K * dv + head + M * K
    return dict(params=params, macs_per_event=macs, config=dict(d=d, modes=n, layers=L, clocks=M, dv=dv))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--result', action='append', required=True, help='dataset=path to a B1 result JSON')
    a = ap.parse_args()
    rows = {}
    for spec in a.result:
        ds, path = spec.split('=', 1)
        o, s = ours(path), s2p2(ds)
        rows[ds] = dict(ours=o, s2p2=s, param_ratio_s2p2_over_ours=s['params'] / o['params'],
                        mac_ratio_s2p2_over_ours=s['macs_per_event'] / o['macs_per_event'])
        print(ds, json.dumps(rows[ds]))
    return rows


if __name__ == '__main__':
    main()
