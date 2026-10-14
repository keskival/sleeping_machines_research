"""Synthetic end-to-end check of the pilot kit: machines emit a cyclic process (8 step types, log-normal step times,
CV 0.15). After the reference date, half of the machines run with one step slowed by --slow. The kit must rank the
faulty machines' monitoring windows above the healthy ones. Reports AUROC per statistic."""
import argparse
import csv
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(); p.add_argument('--slow', type=float, default=1.3); p.add_argument('--out', required=True)
    p.add_argument('--machines', type=int, default=40); p.add_argument('--days', type=int, default=12)
    a = p.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    steps = 8; med = np.exp(rng.uniform(math.log(5), math.log(60), steps))
    ref_until = (a.days - 4) * 86400.
    faulty = set(range(0, a.machines, 2)); slow_step = int(rng.integers(steps))
    with open(out / 'log.csv', 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['timestamp', 'event', 'machine'])
        for mch in range(a.machines):
            t = rng.uniform(0, 60)
            while t < a.days * 86400:
                for s in range(steps):
                    f = a.slow if (mch in faulty and s == slow_step and t > ref_until) else 1.
                    t += med[s] * math.exp(.15 * rng.standard_normal()) * f
                    w.writerow([f'{t:.3f}', f'step{s}', f'm{mch}'])
                t += rng.exponential(120)
    run = out / 'run'
    subprocess.run([sys.executable, str(ROOT / 'experiments/pilot/event_anomaly.py'), '--csv', str(out / 'log.csv'),
                    '--out', str(run), '--stream-col', 'machine', '--window-s', '21600', '--reference-until', str(ref_until),
                    '--epochs', '25', '--patience', '6'], check=True)
    rows = list(csv.DictReader(open(run / 'scores.csv')))
    y = np.array([int(r['stream'][1:]) in faulty for r in rows])
    from sklearn.metrics import roc_auc_score
    res = {k: float(roc_auc_score(y, [float(r[k]) for r in rows])) for k in ('mean_nll', 'glr_max')}
    drv = [r['driving_event'] for r, f in zip(rows, y) if f and r['alarm_glr_max'] == '1']
    res.update(slowed_step=f'step{slow_step}', glr_alarm_rate_faulty=float(np.mean([r['alarm_glr_max'] == '1' for r, f in zip(rows, y) if f])),
               glr_alarm_rate_healthy=float(np.mean([r['alarm_glr_max'] == '1' for r, f in zip(rows, y) if not f])),
               driver_is_slowed_step=float(np.mean([d == f'step{slow_step}' for d in drv])) if drv else None)
    (out / 'check.json').write_text(json.dumps(res, indent=1)); print(json.dumps(res))


if __name__ == '__main__':
    main()
