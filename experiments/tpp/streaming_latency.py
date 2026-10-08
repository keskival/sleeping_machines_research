#!/usr/bin/env python3
"""Measured streaming inference cost of the race-of-clocks model (race_tpp_v5; Taxi checkpoint).

Deployment updates the model one event at a time. This script implements that incremental step (each temporal memory
layer keeps its complex state; the addressed mark memory keeps its slots), checks that the incremental clock parameters
equal the batch encoder's to numerical precision on held-out sequences, and times the per-event update plus next-event
distribution on one CPU thread (batch 1, float64 and float32), after warm-up. Reports microseconds per event and events
per second beside the analytic multiply-accumulate count.
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from race_tpp_v5 import RaceTPP, load_split  # noqa: E402


class Stream:
    """Per-event incremental state for RaceTPP (eval mode)."""

    def __init__(self, model):
        self.m = model; self.reset()

    def reset(self):
        n = self.m.layers[0].n
        self.z = [(torch.zeros(1, n), torch.zeros(1, n)) for _ in self.m.layers]
        self.slots = torch.zeros(1, self.m.K, self.m.marks.dv); self.t_prev = None

    @torch.no_grad()
    def step(self, t, mark):
        m = self.m; dt = 0.0 if self.t_prev is None else max(t - self.t_prev, 0.0) / m.scale; self.t_prev = t
        dtt = torch.tensor([[dt]])
        x = m.embed.weight[mark].view(1, -1) + m.gap(torch.tensor([[math.log1p(dt), float(dt > 0)]]))
        new = []
        for layer, (zr, zi) in zip(m.layers, self.z):
            rate, freq = layer.log_rate.exp(), layer.freq
            w = layer.write(x); g = torch.sigmoid(layer.gate(x)); wr, wi = w[:, :layer.n] * g, w[:, layer.n:] * g
            dec = torch.exp(-rate * dt); ang = freq * dt; cr, ci = dec * torch.cos(ang), dec * torch.sin(ang)
            zr, zi = cr * zr - ci * zi + wr, cr * zi + ci * zr + wi
            h = layer.norm1(x + layer.read(torch.cat([zr, zi], -1)))
            x = layer.norm2(h + layer.mlp(h)); new.append((zr, zi))
        self.z = new
        v = F.softplus(m.marks.value(x)); decay = torch.exp(-m.marks.log_rate.exp() * dt)
        onehot = F.one_hot(torch.tensor([mark]), m.K).to(x.dtype)
        self.slots = self.slots * decay.view(1, 1, -1) + onehot.unsqueeze(-1) * v.unsqueeze(1)
        return m.clocks(x.unsqueeze(1), self.slots.unsqueeze(1))                   # next-event distribution


def main():
    res = json.loads((ROOT / 'experiments/results/tpp/curie_repro_taxi_v5_s0_20261007T0510Z.json').read_text()); a = res['args']
    torch.set_num_threads(1)
    out = dict(checkpoint=res['checkpoint'], dataset='taxi')
    for dtype in (torch.float64, torch.float32):
        torch.set_default_dtype(dtype)
        train = load_split('taxi', 'train'); test = load_split('taxi', 'test')[:100]
        gaps = np.concatenate([np.diff(t) for t, _ in train]); pos = gaps[gaps > 0]
        qs = np.log(np.quantile(pos, np.linspace(0.1, 0.9, a['n_lognormal']))).tolist()
        model = RaceTPP(res['K'], a['d'], a['modes'], a['layers'], a['n_exp'], a['n_lognormal'], a['dv'], a['dropout'],
                        res['scale'], qs, a.get('floor_cell', 0.0))
        sd = torch.load(ROOT / res['checkpoint']); model.load_state_dict({k: v.to(dtype) for k, v in sd.items()}); model.eval()
        st = Stream(model)
        # exactness: incremental vs batch clock parameters on 10 sequences
        err = 0.0
        with torch.no_grad():
            for ts, ms in test[:10]:
                T = torch.tensor([ts], dtype=dtype); M = torch.tensor([ms]); mask = torch.ones_like(M, dtype=torch.bool)
                h, slots = model.encode(T, M, mask); batch = model.clocks(h, slots)
                st.reset()
                for i in range(len(ms)):
                    inc = st.step(float(ts[i]), int(ms[i]))
                    err = max(err, max(float((u[0, 0] - b[0, i]).abs().max()) for u, b in zip(inc, batch)))
        # timing
        events = [(float(t), int(k)) for ts, ms in test for t, k in zip(ts, ms)]
        st.reset()
        for t, k in events[:200]:
            st.step(t, k)
        st.reset(); t0 = time.perf_counter()
        for t, k in events:
            st.step(t, k)
        el = time.perf_counter() - t0
        name = str(dtype).split('.')[-1]
        out[name] = dict(max_abs_diff_incremental_vs_batch=err, events=len(events), us_per_event=1e6 * el / len(events),
                         events_per_second=len(events) / el)
        print(name, out[name], flush=True)
    out['analytic_macs_per_event'] = dict(ours=20708, s2p2=249856, source='experiments/tpp/work.py; report Part I §4.0')
    p = ROOT / 'experiments/results/tpp/streaming_latency_taxi_v5.json'; p.write_text(json.dumps(out, indent=1) + '\n')


if __name__ == '__main__':
    main()
