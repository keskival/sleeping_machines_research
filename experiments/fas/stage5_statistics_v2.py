"""B3 registered statistics: seed means for selection, seed-averaged ranks for CI.

Native rank averaging is retained from v1. Neural reference ranks are averaged
over ALL their selected seeds too, rather than bootstrapping against seed 0.
Missing primary scores must have the same mask across every supplied arm.
"""
import numpy as np
from scipy.stats import rankdata


def auroc(clean, faulty):
    clean, faulty = np.asarray(clean), np.asarray(faulty)
    if not len(clean) or not len(faulty) or not np.isfinite(clean).all() or not np.isfinite(faulty).all():
        raise ValueError('Two nonempty finite primary-score classes required')
    ranks = rankdata(np.r_[clean, faulty])
    n = len(faulty)
    return float((ranks[len(clean):].sum()-n*(n+1)/2)/(len(clean)*n))


def average_ranks(pairs):
    nc = len(pairs[0][0])
    ranks = np.mean([rankdata(np.r_[c, f]) for c, f in pairs], axis=0)
    return ranks[:nc], ranks[nc:]


def analyze(native, baselines, resamples=10000):
    if len(native) != 3 or not baselines or resamples < 1:
        raise ValueError('Three native seeds and eligible references required')
    all_pairs = native + [pair for family in baselines.values() for pair in family]
    if any(len(pair) != 2 for pair in all_pairs):
        raise ValueError('Clean/faulty pair required')
    masks = [~np.isnan(np.asarray(v)) for v in all_pairs[0]]
    if any(not mask.any() for mask in masks):
        raise ValueError('No eligible primary-prefix examples')
    for pair in all_pairs:
        for values, mask in zip(pair, masks):
            values = np.asarray(values)
            if values.ndim != 1 or values.shape != mask.shape or not np.array_equal(~np.isnan(values), mask):
                raise ValueError('Primary-score sample lengths/missing masks differ')
            if not np.isfinite(values[mask]).all():
                raise ValueError('Nonfinite eligible primary scores')
    def retain(pair):
        return tuple(np.asarray(v)[mask] for v, mask in zip(pair, masks))
    native = [retain(pair) for pair in native]
    baselines = {name: [retain(pair) for pair in pairs] for name, pairs in baselines.items()}
    native_auc = [auroc(*pair) for pair in native]
    base_auc = {name: [auroc(*pair) for pair in pairs] for name, pairs in baselines.items()}
    strongest = max(base_auc, key=lambda name: np.mean(base_auc[name]))
    reference_mean = float(np.mean(base_auc[strongest]))
    nc, nf = average_ranks(native)
    bc, bf = average_ranks(baselines[strongest])
    rng = np.random.default_rng(0); differences = []
    for _ in range(resamples):
        ic = rng.integers(0, len(nc), len(nc)); jf = rng.integers(0, len(nf), len(nf))
        differences.append(auroc(nc[ic], nf[jf])-auroc(bc[ic], bf[jf]))
    lo, hi = map(float, np.percentile(differences, [2.5, 97.5]))
    difference = float(np.mean(native_auc)-reference_mean)
    c1, c2, c3 = difference >= .02, lo > 0, all(x > reference_mean for x in native_auc)
    verdict = 'win' if c1 and c2 and c3 else ('loss' if difference <= -.02 and hi < 0 else 'tie')
    return dict(native_seed_aucs=native_auc, native_mean=float(np.mean(native_auc)),
                strongest_baseline=strongest, strongest_auroc=reference_mean, difference=difference,
                bootstrap_95=[lo, hi], criteria=dict(margin_ge_0_02=c1, bootstrap_lower_gt_0=c2,
                                                    every_seed_above=c3), verdict=verdict,
                baselines={name: dict(auroc=float(np.mean(values)), seeds=len(values), seed_aucs=values)
                           for name, values in base_auc.items()},
                eligible_samples=dict(clean=int(masks[0].sum()), faulty=int(masks[1].sum()),
                                      excluded_short_clean=int((~masks[0]).sum()),
                                      excluded_short_faulty=int((~masks[1]).sum())),
                bootstrap_estimand='Paired clean/faulty resamples of seed-averaged ranks on BOTH sides; '
                                    'family selection and margin use arithmetic means of per-seed AUROC')
