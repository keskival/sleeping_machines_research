#!/usr/bin/env python3
"""B1 restart selection: for each protocol seed, keep the restart with the best DEV LL (never reads TEST to choose).

    python experiments/tpp/b1_restart_select.py --prefix b1_final_amazon_v11r

Files: experiments/results/tpp/<prefix>_s<seed>_r<restart>.json. Prints per-seed choice, TEST mean ± sd of the chosen
models, and writes <prefix>_selected.json.
"""
import argparse, glob, json, re, statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--prefix', required=True); a = ap.parse_args()
    groups = {}
    for f in glob.glob(str(ROOT / f'experiments/results/tpp/{a.prefix}_s*_r*.json')):
        s, r = map(int, re.search(r'_s(\d+)_r(\d+)\.json$', f).groups())
        groups.setdefault(s, []).append(json.loads(Path(f).read_text()))
    chosen = {}
    for s, runs in sorted(groups.items()):
        best = max(runs, key=lambda r: r['dev']['ll'])                       # DEV only
        chosen[s] = dict(tag=best['tag'], restarts=len(runs), dev=best['dev'], test=best.get('test'))
        print(f"seed {s}: restarts {len(runs)} DEV {[round(r['dev']['ll'], 4) for r in runs]} -> {best['tag']} TEST {round(best['test']['ll'], 4) if best.get('test') else None}")
    complete = [c for c in chosen.values() if c['restarts'] == 3 and c['test']]
    if complete:
        v = [c['test']['ll'] for c in complete]
        out = dict(prefix=a.prefix, seeds=len(complete), test_ll_mean=statistics.mean(v),
                   test_ll_sd=statistics.stdev(v) if len(v) > 1 else 0.0,
                   time_ll=statistics.mean(c['test']['time_ll'] for c in complete),
                   mark_ll=statistics.mean(c['test']['mark_ll'] for c in complete), chosen=chosen)
        print(f"complete seeds {len(v)}: TEST {out['test_ll_mean']:.4f} ± {out['test_ll_sd']:.4f}  time {out['time_ll']:.3f}  mark {out['mark_ll']:.3f}")
        (ROOT / f'experiments/results/tpp/{a.prefix}_selected.json').write_text(json.dumps(out, indent=1) + '\n')


if __name__ == '__main__':
    main()
