#!/usr/bin/env python3
"""Contract for race_tpp_b4: (1) the first event is scored from t = 0 with an empty history (start event), (2) no
terminal survival term, (3) per-sequence sums averaged over sequences, (4) data rules on every dataset's TRAIN split."""
import json, math, random, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import race_tpp_b4 as b4, race_tpp_v19 as base
torch.set_default_dtype(torch.float64); torch.manual_seed(0)
K = 3
seqs = [(np.array([0.0, 0.7, 1.1, 2.6]), np.array([K, 0, 2, 1])), (np.array([0.0, 0.3]), np.array([K, 1]))]
model = base.RaceTPP(K + 1, 8, 4, 1, 2, 3, 2, 0.0, 0.5, [math.log(0.2), math.log(0.5), math.log(1.0)], 0.0, 0, None, 4).eval()
with torch.no_grad():
    for p in model.parameters(): p.add_(0.2 * torch.randn_like(p))
    out = b4.per_sequence(model, seqs)
    # manual: each sequence alone, every real event scored once, first gap = t1 - 0
    lt = lm = 0.0
    for t, m in seqs:
        T = torch.tensor(t)[None]; M = torch.tensor(m)[None]; mask = torch.ones_like(M, dtype=torch.bool)
        tl, jl, cache = model.event_terms(T, M, mask)
        assert tl.shape[1] == len(t) - 1, 'every real event is scored'
        assert abs(float(cache[3][0, 0]) - t[1]) < 1e-12, 'first gap measured from 0'
        lt += float(tl.sum()); lm += float((jl - tl).sum())
    want_T, want_M = -lt / len(seqs), -lm / len(seqs)
assert abs(out['L_T'] - want_T) < 1e-10 and abs(out['L_M'] - want_M) < 1e-10, (out, want_T, want_M)
print('adapter PASS: L_T', round(out['L_T'], 6), 'L_M', round(out['L_M'], 6))
for ds in ['lastfm_filtered', 'mooc_filtered', 'github_filtered', 'stack_overflow_filtered', 'wikipedia_filtered', 'mimic2_filtered', 'retweets_filtered']:
    rows = []
    for k in range(5):
        tr, K = b4.load(ds, k, 'train'); g = np.concatenate([np.diff(t) for t, _ in tr]); pos = g[g > 0]
        rows.append((b4.recording_cell(pos), int(b4.n_components(pos))))
    print(f'{ds:24s} K={K:3d} cell/components per split', [(f'{c:.3e}', n) for c, n in rows])
