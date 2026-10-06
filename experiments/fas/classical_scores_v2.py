"""Per-run scores of the information-matched classical detectors on a frozen FAS v2 dataset (B3 Stages 3–5).

Fits the nine anonymous-log detectors of calibrate_v2b.py on clean training logs:
- elapsed, tick_count, gap_z, ngram3;
- gap_quantile, gap_cusum, gap_robust_z;
- order3, timed_ngram.

Scores every run of a split at the native drivers' merged prefixes (baselines.PREFIXES: 32 … 1024 process events).
At K = 2 the merged 1,024 is the primary per-line N* = 512. Output: results/fas/<tag>_scores.npz (`<detector>_clean`
and `<detector>_faulty`, runs × prefixes; fault kinds; prefixes) plus <tag>.json with AUROCs.

Test mode (`--split test`) is Stage 4 only and requires `--ledger-reason`. Each scoring appends a line to
results/fas/fas_v2_test_ledger.jsonl (tag, source hashes, UTC time). Classical detectors are deterministic and are
scored on test once.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from baselines import PREFIXES, GapZ, Ngram, auroc, prefix_end, runs  # noqa: E402
from strong_baselines import GapQuantile  # noqa: E402
from timed_ngram import TimedNgram  # noqa: E402

ROOT = HERE.parents[1]
DETECTORS = ('elapsed', 'tick_count', 'gap_z', 'ngram3', 'gap_quantile', 'gap_cusum', 'gap_robust_z', 'order3',
             'timed_ngram')


def fit(train):
    gz = GapZ().fit(train); ng = Ngram().fit(train); gq = GapQuantile().fit(train); tn = TimedNgram().fit(train)
    return dict(elapsed=lambda i, t: float(t[-1]), tick_count=lambda i, t: float((i == 0).sum()), gap_z=gz.score,
                ngram3=ng.score, gap_quantile=gq.quantile, gap_cusum=gq.cusum, gap_robust_z=gq.robust_z,
                order3=tn.order3, timed_ngram=tn.joint)


def score_runs(fns, data):
    out = {name: np.full((len(data), len(PREFIXES)), np.nan) for name in fns}
    for r, (ids, t) in enumerate(data):
        for j, n in enumerate(PREFIXES):
            e = prefix_end(ids, n)
            if e:
                for name, fn in fns.items():
                    out[name][r, j] = fn(ids[:e], t[:e])
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True); p.add_argument('--split', choices=('val', 'test'), default='val')
    p.add_argument('--fit-runs', type=int, default=2000); p.add_argument('--tag', required=True)
    p.add_argument('--ledger-reason', default='', help='required with --split test (Stage 4 confirmation scoring)')
    a = p.parse_args()
    out = ROOT / 'experiments/results/fas' / f'{a.tag}.json'
    if out.exists():
        raise ValueError('Unique unused tag required')
    if a.split == 'test' and not a.ledger_reason:
        raise ValueError('test scoring is Stage 4 only: give --ledger-reason')
    d = ROOT / 'experiments/data/fas' / a.data
    train, _ = runs(d / 'train_clean.npz'); train = train[:a.fit_runs]
    clean, _ = runs(d / f'{a.split}_clean.npz'); faulty, kinds = runs(d / f'{a.split}_faulty.npz')
    fns = fit(train)
    sc_c, sc_f = score_runs(fns, clean), score_runs(fns, faulty)
    arrays = dict(prefixes=np.array(PREFIXES), fault_kind=np.asarray(kinds))
    result = dict(status='completed', battle='B3', data=a.data, split=a.split, fit_runs=len(train), prefixes=PREFIXES,
                  auroc={}, detectors=list(DETECTORS),
                  source_sha256={n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in
                                 ('experiments/fas/classical_scores_v2.py', 'experiments/fas/baselines.py',
                                  'experiments/fas/strong_baselines.py', 'experiments/fas/timed_ngram.py')})
    for name in DETECTORS:
        arrays[f'{name}_clean'] = sc_c[name]; arrays[f'{name}_faulty'] = sc_f[name]
        result['auroc'][name] = {str(n): auroc(sc_c[name][:, j], sc_f[name][:, j]) for j, n in enumerate(PREFIXES)}
    score_path = out.with_name(f'{a.tag}_scores.npz'); np.savez_compressed(score_path, **arrays)
    result['per_run_scores'] = dict(path=str(score_path.relative_to(ROOT)), sha256=hashlib.sha256(score_path.read_bytes()).hexdigest())
    out.write_text(json.dumps(result, indent=1) + '\n')
    if a.split == 'test':
        ledger = ROOT / 'experiments/results/fas/fas_v2_test_ledger.jsonl'
        with ledger.open('a') as fh:
            fh.write(json.dumps(dict(tag=a.tag, kind='classical detectors (deterministic)', data=a.data,
                                     source_sha256=result['source_sha256'], scores_sha256=result['per_run_scores']['sha256'],
                                     reason=a.ledger_reason,
                                     utc=datetime.datetime.now(datetime.UTC).isoformat(timespec='seconds'))) + '\n')
    print(json.dumps({k: v['1024'] for k, v in result['auroc'].items()}))


if __name__ == '__main__':
    main()
