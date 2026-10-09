#!/usr/bin/env python3
"""B10 data preparation (no model fitting): convert each checksum-verified BTCUSDT aggTrades day (1 Aug - 7 Sep 2026) into
compact arrays and summarise the stream.

Per event: t (µs, int64), mark (int8: 3 * side + move; side 0 = buy aggressor, 1 = sell aggressor (buyer is maker);
move 0 = price up, 1 = unchanged, 2 = down vs the previous aggTrade of the day; the first event of a day counts as
unchanged), price (float64), qty (float32). Output: data/binance/npz/<day>.npz and
experiments/results/market/b10_data_summary.json (events per day, zero-gap fraction, gap quantiles, mark frequencies).
"""
import datetime as dt
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / 'data/binance'; OUT = RAW / 'npz'
DAYS = [dt.date(2026, 8, 1) + dt.timedelta(d) for d in range(38)]          # 1 Aug .. 7 Sep 2026


def verify(z):
    want = (RAW / (z.name + '.CHECKSUM')).read_text().split()[0]
    got = hashlib.sha256(z.read_bytes()).hexdigest()
    assert got == want, f'checksum mismatch {z.name}'


def convert(day):
    z = RAW / f'BTCUSDT-aggTrades-{day}.zip'; verify(z)
    with zipfile.ZipFile(z) as zf:
        raw = zf.read(zf.namelist()[0])
    raw = raw.replace(b'True', b'1').replace(b'False', b'0')
    a = np.loadtxt(io.BytesIO(raw), delimiter=',', usecols=(1, 2, 5, 6), dtype=np.float64); del raw
    t = a[:, 2].astype(np.int64)
    if t[0] < 10 ** 14:
        t = t * 1000                                                              # milliseconds in older files
    p, q, sell = a[:, 0], a[:, 1].astype(np.float32), a[:, 3] > 0.5
    order = np.argsort(t, kind='stable'); t, p, q, sell = t[order], p[order], q[order], sell[order]
    dp = np.diff(p, prepend=p[0]); move = np.where(dp > 0, 0, np.where(dp < 0, 2, 1))
    mark = (3 * sell.astype(np.int64) + move).astype(np.int8)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f'{day}.npz', t=t, mark=mark, p=p, q=q)
    g = np.diff(t)
    pos = g[g > 0]
    return dict(day=str(day), events=int(len(t)), zero_gap_frac=float((g == 0).mean()),
                gap_us_quantiles={str(k): float(np.quantile(pos, k)) for k in (0.1, 0.5, 0.9, 0.99)},
                mark_freq=np.bincount(mark, minlength=6).tolist())


def main():
    rows = []
    for day in DAYS:
        rows.append(convert(day)); print(json.dumps(rows[-1]), flush=True)
    out = ROOT / 'experiments/results/market'; out.mkdir(parents=True, exist_ok=True)
    (out / 'b10_data_summary.json').write_text(json.dumps(dict(status='completed', battle='B10', days=rows), indent=1) + '\n')


if __name__ == '__main__':
    sys.exit(main())
