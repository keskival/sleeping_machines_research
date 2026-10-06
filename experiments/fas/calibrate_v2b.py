"""FAS v2 Stage 1 setting calibration, amended form (calibrate_v2.py plus two declared changes; protocol amendment of
6 October 2026, after the original grid's first settings).

Changes from calibrate_v2.py, each from a stated principle, applied to every setting:
1. Per-line prefixes: N counts process events per line, so the merged prefix is N*K (primary N* = 512 per line). The
   fault sits on one line. A fixed merged prefix gives every detector less evidence about that line as K grows, which
   confounds interleaving difficulty with data volume.
2. Line-aware oracle: the identity oracle keys its clean duration statistics by (line, prev type, type). True
   identity includes the line. With a speed offset, pooled statistics blur a 5% line difference that is as large as
   the faults. oracle_max_step (pooled) stays reported.
The gate uses oracle_line_max_step. The selection rule, thresholds, grid and every classical reference are unchanged.
No learned model is involved.

Original description follows.
FAS v2 Stage 1 setting calibration (FAS_V2_CONFIRMATORY_PROTOCOL.md): validation only, oracle and classical only.

For each grid setting (K lines, event dropout p, speed offset delta), generate_v2.sample builds --fit clean training
logs and --val clean + --val faulty validation logs (the future dataset's own train/val seeds; test seeds are never
generated). No learned model is run.

Detectors at prefixes N of process events (faulty = positive, fit on clean training logs only):
- oracle (true line+item identity, oracle_bound scoring): oracle_max_step is the gate's oracle (the protocol's v1
  figures 0.755 / 0.913 are max_step); oracle_sq and oracle_signed are reported.
- oracle-assisted diagnostics, reported but never in the gate: FIFO and the timing-aware tracker, whose route and gap
  statistics come from hidden TRAIN identities.
- information-matched classical references (anonymous logs only): elapsed, tick_count, gap_z, ngram3, gap_quantile,
  gap_cusum, gap_robust_z, order3, timed_ngram.
The structure-assisted beam diagnostic (beam_deinterleave.py) is excluded (THEORY §432).

Selection rule (protocol): at N* = 512, the mildest setting (smallest K, then p, then delta = 0) with
oracle_max_step >= 0.70 and oracle_max_step - best information-matched classical >= 0.08.
Output: experiments/results/fas/fas_v2_calibration_<tag>.json (full grid, either way).
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import generate_v2 as V2  # noqa: E402
import oracle_bound as O  # noqa: E402
import deinterleave_baseline as DI  # noqa: E402
import v2_probe as P  # noqa: E402
from baselines import GapZ, Ngram, auroc, prefix_end  # noqa: E402
from strong_baselines import GapQuantile  # noqa: E402
from timed_ngram import TimedNgram  # noqa: E402

ROOT = HERE.parents[1]
PREFIXES = [128, 256, 512, 960]        # per line; each line keeps >= ~1,007 process events at drop .05
INFO_MATCHED = ('elapsed', 'tick_count', 'gap_z', 'ngram3', 'gap_quantile', 'gap_cusum', 'gap_robust_z', 'order3',
                'timed_ngram')


def line_keyed(ids, t, ident, line):
    """own durations keyed by (line, prev type, type)."""
    return [(k, (int(line[k]),) + key, d) for k, key, d in O.own_durations(ids, t, ident)]


def identity(line, item):
    return np.where(item >= 0, line.astype(np.int64) * 1000 + item, -1)


def oracle_scores(ids, t, ident, stats, n):
    pos = np.flatnonzero(ids != 0)
    if len(pos) < n:
        return (np.nan,) * 3
    end = pos[n - 1] + 1; z = []; per = defaultdict(list)
    for _, key, d in O.own_durations(ids[:end], t[:end], ident[:end]):
        if key in stats:
            m, s = stats[key]; v = float(np.clip((d - m) / s, -50, 50)); z.append(v); per[key].append(v)
    if not z:
        return 0., 0., 0.
    z = np.asarray(z)
    return float((z ** 2).mean()), float(z.mean()), float(max(np.mean(v) for v in per.values()))


def oracle_line_scores(ids, t, ident, line, stats, n):
    pos = np.flatnonzero(ids != 0)
    if len(pos) < n:
        return np.nan
    end = pos[n - 1] + 1; per = defaultdict(list)
    for _, key, d in line_keyed(ids[:end], t[:end], ident[:end], line[:end]):
        if key in stats:
            m, s = stats[key]; per[key].append(float(np.clip((d - m) / s, -50, 50)))
    return float(max(np.mean(v) for v in per.values())) if per else 0.


def robust(acc):
    return {k: (np.median(v), max(1.4826 * np.median(np.abs(np.asarray(v) - np.median(v))), 1.0)) for k, v in acc.items()}


def setting(a, K, drop, delta):
    t0 = time.perf_counter()
    gen = lambda split, n, f: [V2.sample(a.seed_base, split, r, K, drop, delta, f) for r in range(n)]
    fit = gen('train_clean', a.fit, False); clean = gen('val_clean', a.val, False); faulty = gen('val_faulty', a.val, True)
    kinds = np.array([s[4] for s in faulty]); gen_s = time.perf_counter() - t0
    anon = [(s[0].astype(np.int64), s[1]) for s in fit]
    # oracle and oracle-assisted statistics (hidden TRAIN identity)
    acc = defaultdict(list); routes = Counter()
    for ids, t, line, item, *_ in fit:
        ident = identity(line, item)
        for _, key, d in O.own_durations(ids, t, ident):
            acc[key].append(d)
        for i in np.unique(ident[ident >= 0]):
            routes[tuple(int(x) for x in ids[ident == i])] += 1
    stats = robust(acc); route = max(routes, key=routes.get)
    acc_line = defaultdict(list)
    for ids, t, line, item, *_ in fit:
        for _, key, d in line_keyed(ids, t, identity(line, item), line):
            acc_line[key].append(d)
    stats_line = robust(acc_line)
    fifo_acc = defaultdict(list)
    for ids, t, *_ in fit:
        for _, key, d in O.own_durations(ids, t, DI.deinterleave(ids, route)):
            fifo_acc[key].append(d)
    fifo_stats = robust(fifo_acc)
    timed_acc = defaultdict(list)
    for ids, t, *_ in fit:
        for _, key, d in O.own_durations(ids, t, P.deinterleave_timed(ids, t, route, stats)):
            timed_acc[key].append(d)
    timed_stats = robust(timed_acc)
    # information-matched classical references (anonymous logs)
    gz = GapZ().fit(anon); ng = Ngram().fit(anon); gq = GapQuantile().fit(anon); tn = TimedNgram().fit(anon)
    anon_fns = dict(elapsed=lambda i, t: float(t[-1]), tick_count=lambda i, t: float((i == 0).sum()), gap_z=gz.score,
                    ngram3=ng.score, gap_quantile=gq.quantile, gap_cusum=gq.cusum, gap_robust_z=gq.robust_z,
                    order3=tn.order3, timed_ngram=tn.joint)

    def score_all(s):
        ids, t, line, item = s[0].astype(np.int64), s[1], s[2], s[3]
        row = {}
        ident = identity(line, item)
        fifo = DI.deinterleave(ids, route); timed = P.deinterleave_timed(ids, t, route, stats)
        for n_line in PREFIXES:
            n = n_line * K
            sq, sg, mx = oracle_scores(ids, t, ident, stats, n)
            row[('oracle_line_max_step', n_line)] = oracle_line_scores(ids, t, ident, line, stats_line, n)
            row[('oracle_sq', n_line)], row[('oracle_signed', n_line)], row[('oracle_max_step', n_line)] = sq, sg, mx
            row[('fifo_max_step', n_line)] = oracle_scores(ids, t, fifo, fifo_stats, n)[2]
            row[('timed_max_step', n_line)] = oracle_scores(ids, t, timed, timed_stats, n)[2]
            e = prefix_end(ids, n)
            for name, fn in anon_fns.items():
                row[(name, n_line)] = fn(ids[:e], t[:e]) if e else np.nan
        return row

    sc, sf = [score_all(s) for s in clean], [score_all(s) for s in faulty]
    out = defaultdict(dict)
    for key in sc[0]:
        neg = np.array([r[key] for r in sc]); pos = np.array([r[key] for r in sf])
        row = dict(all=auroc(neg, pos))
        for k, kname in ((1, 'wear_and_tear'), (2, 'retry_delay')):
            row[kname] = auroc(neg, pos[kinds == k])
        out[key[0]][key[1]] = row
    best = max(INFO_MATCHED, key=lambda m: out[m][a.n_star]['all'])
    o = out['oracle_line_max_step'][a.n_star]['all']; c = out[best][a.n_star]['all']
    return dict(K=K, drop=drop, delta=delta, auroc=out, gate=dict(oracle=o, best_classical=best, best_classical_auroc=c,
                gap=o - c, qualifies=bool(o >= .70 and o - c >= .08)),
                process_events_mean=float(np.mean([(s[0] != 0).sum() for s in clean])),
                route_share=routes[route] / sum(routes.values()), wall_s=time.perf_counter() - t0, gen_s=gen_s)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tag', required=True); p.add_argument('--seed-base', type=int, default=20261005)
    p.add_argument('--fit', type=int, default=1000); p.add_argument('--val', type=int, default=1000)
    p.add_argument('--K', default='2,3,4'); p.add_argument('--drop', default='0.02,0.05'); p.add_argument('--delta', default='0,0.05')
    p.add_argument('--n-star', type=int, default=512)
    a = p.parse_args()
    out = ROOT / 'experiments/results/fas' / f'fas_v2_calibration_amended_{a.tag}.json'
    if out.exists():
        raise ValueError('Unique unused tag required')
    grid = [(K, d, e) for K in map(int, a.K.split(',')) for d in map(float, a.drop.split(',')) for e in map(float, a.delta.split(','))]
    result = dict(status='running', args=vars(a), protocol='experiments/FAS_V2_CONFIRMATORY_PROTOCOL.md',
                  source_sha256={n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in
                                 ('experiments/fas/calibrate_v2b.py', 'experiments/fas/calibrate_v2.py', 'experiments/fas/generate_v2.py',
                                  'experiments/fas/oracle_bound.py', 'experiments/fas/baselines.py',
                                  'experiments/fas/strong_baselines.py', 'experiments/fas/timed_ngram.py',
                                  'experiments/fas/deinterleave_baseline.py', 'experiments/fas/v2_probe.py',
                                  'experiments/FAS_V2_CONFIRMATORY_PROTOCOL.md')},
                  prefix_unit='process events per line (merged prefix = N * K)',
                  classes=dict(oracle=['oracle_line_max_step (gate)', 'oracle_max_step', 'oracle_sq', 'oracle_signed'],
                               oracle_assisted_diagnostics=['fifo_max_step', 'timed_max_step'],
                               information_matched=list(INFO_MATCHED),
                               excluded=['beam_deinterleave.py (structure-assisted diagnostic, THEORY §432)']),
                  settings=[])
    for K, d, e in grid:
        r = setting(a, K, d, e); result['settings'].append(r)
        print(json.dumps(dict(K=K, drop=d, delta=e, **r['gate'], wall_s=round(r['wall_s']))), flush=True)
        tmp = out.with_suffix('.partial.json'); tmp.write_text(json.dumps(result, indent=1, default=float) + '\n')
    order = sorted(result['settings'], key=lambda s: (s['K'], s['drop'], s['delta'] != 0, s['delta']))
    chosen = next((s for s in order if s['gate']['qualifies']), None)
    result['selection'] = (dict(K=chosen['K'], drop=chosen['drop'], delta=chosen['delta'], gate=chosen['gate'])
                           if chosen else 'no setting qualifies: FAS in this form is not a discriminating home-field benchmark')
    result['status'] = 'completed'
    out.write_text(json.dumps(result, indent=1, default=float) + '\n')
    out.with_suffix('.partial.json').unlink(missing_ok=True)
    print(json.dumps(dict(selection=result['selection']), default=float))


if __name__ == '__main__':
    main()
