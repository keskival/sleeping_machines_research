#!/usr/bin/env python3
"""B1 scoreboard: aggregate completed final-protocol seeds (TEST, best-DEV checkpoint) against published references.

Reads experiments/results/tpp/b1_final_<dataset>_<version>_s<seed>.json; reports mean and sample sd over seeds of
total/time/mark per-event LL, RMSE and accuracy, and the gap to the best published model (S2P2 Table 8; 5 seeds).
A win requires the mean to exceed the best published mean; it is reported with both seed spreads.
"""
import glob, json, re, statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REF = {  # dataset: (best model, total, sd, best time, best mark) from S2P2 Table 8 / HHP Table 2
    'amazon': ('S2P2', 0.781, 0.011, 2.652, -1.871),
    'retweet': ('NHP', -6.348, 0.000, -5.584, -0.764),
    'taxi': ('S2P2', 0.522, 0.004, 0.735, -0.211),
    'taobao': ('IFTPP', 1.318, 0.017, 2.719, -1.391),
    'stackoverflow': ('S2P2', -2.163, 0.009, -0.641, -1.510),
}


def main():
    out = {}
    for ds, (name, ref, ref_sd, ref_time, ref_mark) in REF.items():
        runs = [json.loads(Path(f).read_text()) for f in sorted(glob.glob(str(ROOT / f'experiments/results/tpp/b1_final_{ds}_*_s*.json')))]
        runs = [r for r in runs if 'test' in r]
        if not runs:
            continue
        agg = {}
        for k in ('ll', 'time_ll', 'mark_ll', 'rmse', 'acc'):
            v = [r['test'][k] for r in runs]
            agg[k] = (statistics.mean(v), statistics.stdev(v) if len(v) > 1 else 0.0)
        out[ds] = dict(seeds=len(runs), version=sorted({re.sub(r'_s\d+$', '', r['tag']) for r in runs}),
                       params=runs[0]['parameters'], test={k: dict(mean=m, sd=s) for k, (m, s) in agg.items()},
                       reference=dict(model=name, ll=ref, sd=ref_sd, best_time_ll=ref_time, best_mark_ll=ref_mark),
                       gap_ll=agg['ll'][0] - ref, win=agg['ll'][0] > ref)
        m, s = agg['ll']
        print(f"{ds:14} n={len(runs)}  ours {m:.4f} ± {s:.4f}  vs {name} {ref:.3f} ± {ref_sd:.3f}  gap {m - ref:+.4f}"
              f"  time {agg['time_ll'][0]:.3f} (best {ref_time})  mark {agg['mark_ll'][0]:.3f} (best {ref_mark})"
              f"  {'WIN' if m > ref else 'behind'}")
    (ROOT / 'experiments/results/tpp/b1_scoreboard.json').write_text(json.dumps(out, indent=1) + '\n')


if __name__ == '__main__':
    main()
