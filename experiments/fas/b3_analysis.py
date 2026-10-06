"""B3 Stage 5 analysis and decision (FAS_V2_CONFIRMATORY_PROTOCOL.md, with amendments). Mechanical: the inputs are per-run
score files, and the rule is fixed.

Spec (JSON):
{"data": "<dataset>", "split": "test" | "val", "n_star": 1024,
 "native": {"seeds": {"6": "<scores.npz>", ...}, "rule": "total"},
 "neural": [{"name": "lstm", "seeds": {"0": "<scores.npz>", ...}, "rule": "total"}, ...],
 "classical": {"path": "<classical scores.npz>"},
 "val": {"native": {...}, "neural": [...], "classical": {...}}}      # validation scores, for alarm calibration only

Learned score files (native.py, native_race_readout.py, dense.py) hold <split>_clean_<rule> and <split>_faulty_<rule>
(runs × prefixes). Classical files (classical_scores_v2.py) hold <detector>_clean / <detector>_faulty. Prefixes are
merged process events (baselines.PREFIXES); at K = 2, merged 1,024 = N* = 512 per line.

Decision (Stage 5):
- strongest baseline = the highest test AUROC at N* among the classical detectors and the neural families' seed means;
- WIN if all three hold:
  1. native seed-mean AUROC − strongest ≥ 0.02;
  2. the paired bootstrap (10,000 resamples, stratified clean/faulty; native scores averaged over seeds, each seed's
     scores standardised on its clean scores) gives a 95% interval for AUROC(native) − AUROC(strongest) with lower
     bound > 0;
  3. every native seed's AUROC exceeds the strongest baseline's point estimate.
- otherwise TIE if |difference| < 0.02 or the interval covers 0, else LOSS.

Operational metric: for each method, per-prefix alarm thresholds sit at a common upper quantile of validation-clean
scores, chosen so that the any-prefix false-alarm rate on validation-clean is 1%. On the evaluated split this reports
the detection rate of faulty runs, the median simulated minutes to the first alarm (time of the alarming prefix's last
process event) and the realised clean false-alarm rate.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from baselines import PREFIXES  # noqa: E402


def auc(neg, pos):
    s = np.concatenate([neg, pos]); r = rankdata(s)
    return float((r[len(neg):].sum() - len(pos) * (len(pos) + 1) / 2) / (len(neg) * len(pos)))


def standardise(clean, faulty):
    mu, sd = np.nanmean(clean), np.nanstd(clean) + 1e-12
    return (clean - mu) / sd, (faulty - mu) / sd


def load_learned(paths, split, rule):
    """per seed: (clean (n, P), faulty (n, P)) arrays."""
    out = {}
    for seed, path in paths.items():
        z = np.load(ROOT / path if not Path(path).is_absolute() else path)
        out[seed] = (z[f'{split}_clean_{rule}'], z[f'{split}_faulty_{rule}'])
    return out


def load_classical(path):
    z = np.load(ROOT / path if not Path(path).is_absolute() else path)
    names = sorted({k[:-6] for k in z.files if k.endswith('_clean')})
    return {n: (z[f'{n}_clean'], z[f'{n}_faulty']) for n in names}


def seed_average(seeds):
    cs, fs = zip(*[standardise(c, f) for c, f in seeds.values()])
    return np.mean(cs, 0), np.mean(fs, 0)


def bootstrap_diff(a, b, B, rng):
    """a, b: (clean, faulty) score vectors for the same runs. Returns the B stratified resampled AUROC(a) - AUROC(b)."""
    (ac, af), (bc, bf) = a, b
    nc, nf = len(ac), len(af); out = np.empty(B)
    for i in range(B):
        ic, jf = rng.integers(0, nc, nc), rng.integers(0, nf, nf)
        out[i] = auc(ac[ic], af[jf]) - auc(bc[ic], bf[jf])
    return out


def bootstrap_ci(a, B, rng):
    ac, af = a; vals = np.empty(B)
    for i in range(B):
        vals[i] = auc(ac[rng.integers(0, len(ac), len(ac))], af[rng.integers(0, len(af), len(af))])
    return [float(np.quantile(vals, .025)), float(np.quantile(vals, .975))]


def alarm_thresholds(val_clean, far=.01):
    """common upper quantile q per prefix so that P(any prefix exceeds) = far on validation-clean."""
    lo, hi = 1 - far, 1.0
    for _ in range(60):
        q = (lo + hi) / 2
        th = np.nanquantile(val_clean, q, axis=0)
        rate = np.mean(np.any(val_clean > th[None], 1))
        lo, hi = (q, hi) if rate > far else (lo, q)
    return np.nanquantile(val_clean, hi, axis=0)


def prefix_minutes(data, split):
    z = np.load(ROOT / 'experiments/data/fas' / data / f'{split}_faulty.npz')
    o, ids, t = z['offsets'], z['ids'], z['times_ms']
    out = np.full((len(o) - 1, len(PREFIXES)), np.nan)
    for r in range(len(o) - 1):
        tt = t[o[r]:o[r + 1]][ids[o[r]:o[r + 1]] != 0]
        for j, n in enumerate(PREFIXES):
            if n <= len(tt):
                out[r, j] = tt[n - 1] / 60000.
    return out


def operational(clean, faulty, val_clean, minutes):
    th = alarm_thresholds(val_clean)
    fire_f = faulty > th[None]; fire_c = clean > th[None]
    detected = fire_f.any(1); first = np.argmax(fire_f, 1)
    mins = minutes[np.arange(len(first)), first][detected]
    return dict(detection_rate=float(detected.mean()), median_minutes_to_alarm=float(np.median(mins)) if len(mins) else None,
                clean_false_alarm_rate=float(fire_c.any(1).mean()))


def decide(native_seed_aucs, native_avg, strongest, B=10000, seed=0, margin=.02):
    rng = np.random.default_rng(seed)
    mean_auc = float(np.mean(list(native_seed_aucs.values())))
    diffs = bootstrap_diff(native_avg, strongest['scores'], B, rng)
    lo, hi = float(np.quantile(diffs, .025)), float(np.quantile(diffs, .975))
    d = mean_auc - strongest['auroc']
    c1, c2 = d >= margin, lo > 0
    c3 = all(v > strongest['auroc'] for v in native_seed_aucs.values())
    verdict = 'WIN' if (c1 and c2 and c3) else ('TIE' if (abs(d) < margin or lo <= 0 <= hi) else 'LOSS')
    return dict(verdict=verdict, native_mean_auroc=mean_auc, strongest=strongest['name'], strongest_auroc=strongest['auroc'],
                difference=d, paired_bootstrap_ci95=[lo, hi], conditions=dict(margin=c1, ci_lower_gt_0=c2, every_seed_above=c3))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec', required=True); p.add_argument('--tag', required=True)
    p.add_argument('--resamples', type=int, default=10000)
    a = p.parse_args()
    out = ROOT / 'experiments/results/fas' / f'{a.tag}.json'
    if out.exists():
        raise ValueError('Unique unused tag required')
    spec = json.loads(Path(a.spec).read_text()); split = spec['split']; j = PREFIXES.index(spec.get('n_star', 1024))
    native = load_learned(spec['native']['seeds'], split, spec['native'].get('rule', 'total'))
    native_aucs = {s: auc(c[:, j], f[:, j]) for s, (c, f) in native.items()}
    nat_c, nat_f = seed_average(native)
    candidates = []
    for fam in spec.get('neural', []):
        seeds = load_learned(fam['seeds'], split, fam.get('rule', 'total'))
        aucs = {s: auc(c[:, j], f[:, j]) for s, (c, f) in seeds.items()}
        c, f = seed_average(seeds)
        candidates.append(dict(name=fam['name'], kind='neural seed mean', auroc=float(np.mean(list(aucs.values()))),
                               seed_aurocs=aucs, scores=(c[:, j], f[:, j])))
    if spec.get('classical'):
        for name, (c, f) in load_classical(spec['classical']['path']).items():
            candidates.append(dict(name=name, kind='classical', auroc=auc(c[:, j], f[:, j]), scores=(c[:, j], f[:, j])))
    strongest = max(candidates, key=lambda c: c['auroc'])
    result = dict(status='completed', battle='B3', split=split, n_star_merged=PREFIXES[j], spec=spec,
                  native_seed_aurocs=native_aucs,
                  native_seed_avg_ci95=bootstrap_ci((nat_c[:, j], nat_f[:, j]), 2000, np.random.default_rng(1)),
                  references={c['name']: dict(kind=c['kind'], auroc=c['auroc'],
                                              ci95=bootstrap_ci(c['scores'], 2000, np.random.default_rng(2)))
                              for c in candidates},
                  decision=decide(native_aucs, (nat_c[:, j], nat_f[:, j]), strongest, B=a.resamples),
                  native_seed_mean_auroc_by_prefix={str(n): float(np.mean([auc(c[:, jj], f[:, jj]) for c, f in native.values()]))
                                                    for jj, n in enumerate(PREFIXES)},
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if spec.get('val'):
        minutes = prefix_minutes(spec['data'], split)
        vnat = load_learned(spec['val']['native']['seeds'], 'val', spec['val']['native'].get('rule', 'total'))
        v_c, _ = seed_average(vnat)
        result['operational'] = {'native (seed-averaged)': operational(nat_c, nat_f, v_c, minutes)}
        if spec['val'].get('classical'):
            vcl = load_classical(spec['val']['classical']['path']); tcl = load_classical(spec['classical']['path'])
            for name in tcl:
                result['operational'][name] = operational(tcl[name][0], tcl[name][1], vcl[name][0], minutes)
    out.write_text(json.dumps(result, indent=1, default=float) + '\n')
    print(json.dumps(result['decision']))


if __name__ == '__main__':
    main()
