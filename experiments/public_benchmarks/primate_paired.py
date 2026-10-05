"""Paired per-session comparison of the official six-session primate run (P0-3) with the published tinyRSNN/bigRSNN.

Reads the completed `aws_primate_r1_*` result files and the cited reference summaries in references/. Prints per-session
R² and the means over the sessions completed so far. The leaderboard compares six-session means only; AEGRU (.71)
publishes no per-session values.
"""
import glob
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REF = {n: json.loads((ROOT / f'experiments/public_benchmarks/references/fmi_basel_rsnn_results_summary_{n}.json').read_text())
       for n in ('tinyRSNN', 'bigRSNN')}


def table():
    rows = []
    for f in sorted(glob.glob(str(ROOT / 'experiments/results/neurobench_primate/aws_primate_r1_*.json'))):
        s = json.loads(Path(f).read_text())['sessions'][0]
        rows.append(dict(session=s['session'], ours=s['test_r2'], tinyRSNN=REF['tinyRSNN'][s['session']]['r2'],
                         bigRSNN=REF['bigRSNN'][s['session']]['r2']))
    return rows


if __name__ == '__main__':
    rows = table()
    for r in rows:
        print(f"{r['session']:18s} ours {r['ours']:.3f}  tinyRSNN {r['tinyRSNN']:.3f}  bigRSNN {r['bigRSNN']:.3f}")
    if rows:
        n = len(rows); m = {k: sum(r[k] for r in rows) / n for k in ('ours', 'tinyRSNN', 'bigRSNN')}
        print(f"mean over {n}/6 sessions: ours {m['ours']:.3f}  tinyRSNN {m['tinyRSNN']:.3f}  bigRSNN {m['bigRSNN']:.3f}"
              f"  (six-session leaderboard: AEGRU .710, bigRSNN {REF['bigRSNN']['average']['r2']:.3f}, tinyRSNN {REF['tinyRSNN']['average']['r2']:.3f})")
