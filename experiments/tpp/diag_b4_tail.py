#!/usr/bin/env python3
"""B4 diagnosis (evaluation only, no training, no score replacement): per-event time log density of a saved B4
checkpoint on one split's TEST (or VAL) data, to locate extreme tail events. Reports the worst events with their gap,
the gap's rank in the TRAIN gap distribution, the elapsed-time scale, and the survival term. Writes a JSON report."""
import argparse, json, math, random, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import race_tpp_v19 as base  # noqa: E402
from race_tpp_b4 import load, recording_cell, n_components  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--result', required=True); ap.add_argument('--part', default='test')
    ap.add_argument('--out', required=True); ap.add_argument('--top', type=int, default=15)
    a = ap.parse_args(); torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
    r = json.loads((ROOT / a.result).read_text()); cfg = r['config']; ds, sp = r['dataset'], r['split']
    train, K = load(ds, sp, 'train')
    gaps = np.concatenate([np.diff(t) for t, _ in train]); pos = gaps[gaps > 0]
    cell = recording_cell(pos); comps = n_components(pos); n_window = 4 if comps >= 2 else 0
    scale = float(np.median(pos)); qs = np.log(np.quantile(pos, np.linspace(0.1, 0.9, cfg['n_lognormal']))).tolist()
    edges = base.cluster_windows(pos, n_window, r['args']['seed']) if n_window else None
    model = base.RaceTPP(K + 1, cfg['d'], cfg['modes'], cfg['layers'], cfg['n_exp'], cfg['n_lognormal'], cfg['dv'],
                         cfg['dropout'], scale, qs, cell, n_window, edges, cfg['state_modes'])
    model.load_state_dict(torch.load(ROOT / r['checkpoint'])); model.eval()
    seqs, _ = load(ds, sp, a.part); rows = []; per_seq = []
    with torch.no_grad():
        for si, (t, m) in enumerate(seqs):
            tt = torch.tensor(t)[None]; mm = torch.tensor(m)[None]; mask = torch.ones_like(mm, dtype=torch.bool)
            time_lp, joint_lp, _ = model.event_terms(tt, mm, mask)
            lp = time_lp[0].numpy(); g = np.diff(t)
            per_seq.append(float(-lp.sum()))
            for j in range(len(lp)):
                rows.append(dict(seq=si, event=j + 1, nll=float(-lp[j]), gap=float(g[j]), gap_over_scale=float(g[j] / scale),
                                 train_gap_quantile=float((pos <= g[j]).mean()) if g[j] > 0 else 0.0))
    rows.sort(key=lambda x: -x['nll']); ps = np.array(per_seq)
    rep = dict(result=a.result, part=a.part, sequences=len(seqs), mean_L_T=float(ps.mean()), median_seq_L_T=float(np.median(ps)),
               worst_sequences=sorted([(float(v), i) for i, v in enumerate(ps)], reverse=True)[:5], worst_events=rows[:a.top],
               events_nll_over_1e3=int(sum(x['nll'] > 1e3 for x in rows)), train_max_gap_over_scale=float(pos.max() / scale),
               scale=scale, recording_cell=cell)
    Path(a.out).write_text(json.dumps(rep, indent=1) + '\n'); print(json.dumps({k: rep[k] for k in ('mean_L_T', 'median_seq_L_T', 'worst_sequences', 'events_nll_over_1e3', 'train_max_gap_over_scale')}))
    for x in rows[:8]:
        print(x)


if __name__ == '__main__':
    main()
