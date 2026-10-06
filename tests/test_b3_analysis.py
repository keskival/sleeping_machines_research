"""Contracts of the B3 Stage 5 analysis (experiments/fas/b3_analysis.py)."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments/fas'))
import b3_analysis as A  # noqa: E402
from baselines import auroc as reference_auroc  # noqa: E402


def _scores(rng, shift, n=400):
    return rng.normal(0, 1, n), rng.normal(shift, 1, n)


def _avg(seeds):
    """seed-averaged (clean, faulty) vectors at one prefix, as b3_analysis.main builds them."""
    return tuple(x[:, 0] for x in A.seed_average({k: (v[0][:, None], v[1][:, None]) for k, v in seeds.items()}))


def test_auc_matches_reference_with_ties():
    rng = np.random.default_rng(0)
    c, f = np.round(rng.normal(0, 1, 300), 1), np.round(rng.normal(.4, 1, 300), 1)
    assert abs(A.auc(c, f) - reference_auroc(c, f)) < 1e-12


def test_decision_win_tie_loss():
    rng = np.random.default_rng(1)
    base = _scores(rng, .5)
    strongest = dict(name='base', auroc=A.auc(*base), scores=base)
    noise = lambda s: (s[0] + rng.normal(0, .3, len(s[0])), s[1] + rng.normal(0, .3, len(s[1])))
    strong = (base[0], base[1] + 1.0)                 # clearly better on the same runs
    seeds = {k: noise(strong) for k in '678'}
    win = A.decide({k: A.auc(*v) for k, v in seeds.items()}, _avg(seeds), strongest, B=500)
    assert win['verdict'] == 'WIN', win
    same = {k: noise(base) for k in '678'}
    tie = A.decide({k: A.auc(*v) for k, v in same.items()}, _avg(same), strongest, B=500)
    assert tie['verdict'] == 'TIE', tie
    weak = {k: noise((base[0], base[1] - 1.0)) for k in '678'}
    loss = A.decide({k: A.auc(*v) for k, v in weak.items()}, _avg(weak), strongest, B=500)
    assert loss['verdict'] == 'LOSS', loss


def test_every_seed_condition_blocks_a_win():
    rng = np.random.default_rng(2)
    base = _scores(rng, .5); strongest = dict(name='base', auroc=A.auc(*base), scores=base)
    good = (base[0], base[1] + 1.0)
    seeds = {'6': good, '7': good, '8': (base[0], base[1] - .5)}       # one seed below the baseline
    r = A.decide({k: A.auc(*v) for k, v in seeds.items()}, _avg(seeds), strongest, B=300)
    assert not r['conditions']['every_seed_above'] and r['verdict'] != 'WIN'


def test_alarm_thresholds_hit_one_percent_on_validation():
    rng = np.random.default_rng(3)
    val = rng.normal(0, 1, (5000, 6)).cumsum(1)
    th = A.alarm_thresholds(val, .01)
    rate = np.mean(np.any(val > th[None], 1))
    assert abs(rate - .01) < .002, rate
