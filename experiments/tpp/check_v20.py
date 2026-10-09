#!/usr/bin/env python3
"""Contracts for race_tpp_v20: (1) flags off == v19 exactly on a trained checkpoint (B4 MOOC split 0 validation);
(2) flags on, zero-initialized extra term == v19 at load; (3) time log-likelihood has zero gradient into the mark-only
pathway (mark_layers), mark log-likelihood a non-zero one; (4) mark distributions normalize."""
import json, random, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import race_tpp_b4 as b4, race_tpp_v19 as v19, race_tpp_v20 as v20
torch.set_default_dtype(torch.float64)
r = json.load(open('experiments/results/tpp_b4/b4_mooc_s0.json')); c = r['config']
train, K = b4.load('mooc_filtered', 0, 'train'); val, _ = b4.load('mooc_filtered', 0, 'val'); val = val[:200]
g = np.concatenate([np.diff(t) for t, _ in train]); pos = g[g > 0]
qs = np.log(np.quantile(pos, np.linspace(.1, .9, c['n_lognormal']))).tolist(); e = v19.cluster_windows(pos, r['n_window'], 0)
args = (K + 1, c['d'], c['modes'], c['layers'], c['n_exp'], c['n_lognormal'], c['dv'], 0.0, r['scale'], qs, r['recording_cell'], r['n_window'], e, c['state_modes'])
sd = torch.load(r['checkpoint'])
def score(m):
    m.eval(); return b4.per_sequence(m, val)
a = v19.RaceTPP(*args); a.load_state_dict(sd); ref = score(a)
b = v20.RaceTPP(*args); b.load_state_dict(sd); o1 = score(b)
assert all(abs(o1[k] - ref[k]) < 1e-10 for k in ('L_T', 'L_M')), (o1, ref); print('1 flags off == v19 PASS', ref)
cb = v20.RaceTPP(*args, mark_mem=1, mark_stats=True); missing = cb.load_state_dict(sd, strict=False); o2 = score(cb)
assert all(abs(o2[k] - ref[k]) < 1e-10 for k in ('L_T', 'L_M')), (o2, ref); print('2 flags on at load == v19 PASS')
with torch.no_grad():
    cb.mark_extra.weight.normal_(0, 0.1)
cb.train(); t, m, mask = next(b4.base.batches(val, 16, False, random.Random(0)))
tl, jl, _ = cb.event_terms(t, m, mask); v = mask[:, 1:].to(tl.dtype)
gt = torch.autograd.grad((tl * v).sum(), list(cb.mark_layers.parameters()), allow_unused=True, retain_graph=True)
gm = torch.autograd.grad(((jl - tl) * v).sum(), list(cb.mark_layers.parameters()), allow_unused=True)
nt = sum(float(x.abs().sum()) for x in gt if x is not None); nm = sum(float(x.abs().sum()) for x in gm if x is not None)
assert nt == 0.0 and nm > 0, (nt, nm); print('3 time grad into mark pathway', nt, 'mark grad', round(nm, 4), 'PASS')
p = cb.clocks(*[q[:, :-1] for q in cb.encode(t, m, mask)])[4]
assert torch.allclose(p.exp().sum(-1), torch.ones_like(p[..., 0])); print('4 mark normalization PASS')
