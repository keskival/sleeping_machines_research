"""Contracts of the FAS v2 generator (FAS_V2_CONFIRMATORY_PROTOCOL.md, Stage 0)."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments/fas'))
import generate_v2 as V2  # noqa: E402


def synthetic(seed, n, t0=0, step=1000, item_offset=0):
    ids = np.array([0] + [1 + (k % 5) for k in range(n)], np.uint8)
    t = np.array([0] + [t0 + step * k for k in range(n)], np.int64)
    item = np.array([-1] + [item_offset + k // 5 for k in range(n)], np.int64)
    return seed, ids, t, item


def test_merge_keeps_every_process_event_and_one_clock():
    lines = [synthetic(11, 40), synthetic(12, 40, t0=500)]
    ids, t, line, item = V2.merge(lines, 0., 0., 11)
    assert np.all(np.diff(t) >= 0)
    assert (line == 0).sum() == 40 and (line == 1).sum() == 40
    ticks = t[ids == 0]
    assert np.array_equal(ticks, np.arange(0, t[ids != 0].max() + 1, V2.TICK_MS))
    for l, (_, i0, t0, it0) in enumerate(lines):
        assert np.array_equal(t[line == l], t0[1:]) and np.array_equal(ids[line == l], i0[1:])


def test_dropout_depends_only_on_line_seed():
    a = V2.merge([synthetic(7, 400), synthetic(8, 400)], .05, 0., 7)
    b = V2.merge([synthetic(7, 400, item_offset=100), synthetic(8, 400)], .05, 0., 99)
    for l in (0, 1):
        assert np.array_equal(np.sort(a[1][a[2] == l]), np.sort(b[1][b[2] == l]))
    kept = (a[2] >= 0).sum() / 800
    assert .9 < kept < 1.


def test_ties_are_not_line_ordered():
    first = set()
    for s in range(20):
        ids, t, line, _ = V2.merge([synthetic(s, 10), synthetic(s + 100, 10)], 0., 0., s)
        proc = line >= 0
        first.add(int(line[proc][0]))
    assert first == {0, 1}


def test_speed_offset_alternates_by_line():
    ids, t, line, _ = V2.merge([synthetic(1, 10), synthetic(2, 10)], 0., .05, 1)
    assert t[line == 0].max() == round(9000 * 1.05) and t[line == 1].max() == round(9000 * .95)


def test_simulated_sample_is_deterministic_and_faulty_line_zero():
    a = V2.sample(20261005, 'val_faulty', 0, 2, .02, 0., True)
    b = V2.sample(20261005, 'val_faulty', 0, 2, .02, 0., True)
    assert all(np.array_equal(x, y) for x, y in zip(a[:4], b[:4]))
    assert a[4] in (1, 2) and a[5] == [V2.line_seed(20261005, 'val_faulty', 2, 0, l) for l in range(2)]
    c = V2.sample(20261005, 'val_clean', 0, 2, .02, 0., False)
    assert c[4] == 0
