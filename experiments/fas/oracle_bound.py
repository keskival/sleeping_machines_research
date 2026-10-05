"""Identity-aware oracle bound for the FAS benchmark (FAS_BENCHMARK.md backlog, 5 October).

Question: is the early range (N <= 256 process events) information-limited for every model, or is the signal there and
only hidden by the interleaving?  The oracle regenerates the official seeds with item identity recovered from the
simulator (each component call is a SimPy sub-process created inside its item's process; the sub-process -> item map is
recorded at creation).  It verifies that the event ids and times equal the identity-free data exactly, then scores each
item's own step-to-step durations against clean training durations of the same step transition.

Scores over the prefix of N process events (faulty = positive, one-class, fit on clean training seeds only):
  oracle_sq     - mean squared robust z of the item-own durations (two-sided)
  oracle_signed - mean signed robust z (faults add delay)
  oracle_max_step - max over step transitions of the mean signed z of that transition (a localized fault concentrates in
                  one module's steps)
Not a deployable model: it uses identity the benchmark hides.  It bounds what identity-free models could reach.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import generate as G  # noqa: E402
from baselines import PREFIXES, auroc  # noqa: E402

ROOT = HERE.parents[1]


def run_with_identity(seed, faulty, items=30):
    """generate.generate_run with a logger that records the item of every event; returns ids, times, item, kind."""
    env_box = {}
    parent = {}
    orig_descriptor = G.simpy.Environment.__dict__['process']
    Process = G.simpy.events.Process

    def process(self, gen):
        p = Process(self, gen)
        parent[id(p)] = self.active_process          # creator (item process or None)
        p._keep = True
        return p

    lines = []

    class Log:
        def __init__(self):
            self.lines = lines

        def addMessage(self, msgtype, metadata=''):
            env = env_box['env']
            lines.append((env.now, msgtype, env.active_process))

    G.simpy.Environment.process = process
    try:
        real_env = G.simpy.Environment
        class Env(real_env):
            def __init__(self, *a, **k):
                super().__init__(*a, **k); env_box['env'] = self
        G.simpy.Environment = Env
        import random
        random.seed(seed); np.random.seed(seed % 2**32)
        env = Env()
        logger = Log()
        line = G.ProductionLine(env, logger, False)
        G.Clock(env, logger, False).spawn()
        kind = 0
        if faulty:
            kind = random.choice([1, 2])

            def add_fault():
                yield env.timeout(0)
                if kind == 1:
                    module = random.sample(line.conveyors, 1)[0]; module.add_fault(G.WearAndTear(env, module, False))
                else:
                    module = random.sample(line.bowl_feeders, 1)[0]; module.add_fault(G.RetryDelay(env, module, False))
            env.process(add_fault())
        roots = []
        last = None
        for _ in range(items):
            last = G.FASInstance(env, line, logger).spawn(); roots.append(last)
        env.run(last)
    finally:
        G.simpy.Environment = real_env
        G.simpy.Environment.process = orig_descriptor
    root_index = {id(r): i for i, r in enumerate(roots)}

    def item_of(p):
        while p is not None and id(p) not in root_index:
            p = parent.get(id(p))
        return root_index[id(p)] if p is not None else -1

    times = np.array([t for t, _, _ in lines], dtype=np.int64)
    ids = np.array([G.EVENT_ID[m] for _, m, _ in lines], dtype=np.uint8)
    item = np.array([item_of(p) if m != 'TICK' else -1 for _, m, p in lines], dtype=np.int64)
    return ids, times, item, kind


def own_durations(ids, times, item):
    """per process event with a previous event of the same item: ((prev id, id), duration)."""
    last = {}; out = []
    for k in range(len(ids)):
        if ids[k] == 0 or item[k] < 0:
            continue
        i = item[k]
        if i in last:
            pk = last[i]; out.append((k, (int(ids[pk]), int(ids[k])), int(times[k] - times[pk])))
        last[i] = k
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True, help='identity-free dataset to verify against (experiments/data/fas/<name>)')
    p.add_argument('--seed-base', type=int, default=20261004); p.add_argument('--fit-runs', type=int, default=2000)
    p.add_argument('--test-runs', type=int, default=2000); p.add_argument('--out', default='')
    a = p.parse_args()
    split_index = {name: i for i, name in enumerate(G.SPLITS)}
    ref = {s: np.load(ROOT / 'experiments/data/fas' / a.data / f'{s}.npz') for s in ('test_clean', 'test_faulty')}

    def gen(split, n, faulty, verify=None):
        out = []
        for r in range(n):
            seed = a.seed_base + 10_000_000 * split_index[split] + r
            ids, t, item, kind = run_with_identity(seed, faulty)
            if verify is not None:
                o = verify['offsets']
                assert np.array_equal(ids, verify['ids'][o[r]:o[r + 1]]) and np.array_equal(t, verify['times_ms'][o[r]:o[r + 1]]), (split, r)
            out.append((ids, t, item, kind))
        return out

    train = gen('train_clean', a.fit_runs, False)
    acc = defaultdict(list)
    for ids, t, item, _ in train:
        for _, key, d in own_durations(ids, t, item):
            acc[key].append(d)
    stats = {k: (np.median(v), max(1.4826 * np.median(np.abs(np.asarray(v) - np.median(v))), 1.0)) for k, v in acc.items()}
    clean = gen('test_clean', a.test_runs, False, ref['test_clean'])
    faulty = gen('test_faulty', a.test_runs, True, ref['test_faulty'])
    kinds = np.array([k for *_, k in faulty])

    def scores(run, n):
        ids, t, item, _ = run
        pos = np.flatnonzero(ids != 0)
        if len(pos) < n:
            return (np.nan,) * 3
        end = pos[n - 1] + 1
        z = []; per = defaultdict(list)
        for k, key, d in own_durations(ids[:end], t[:end], item[:end]):
            if key in stats:
                m, s = stats[key]; v = float(np.clip((d - m) / s, -50, 50)); z.append(v); per[key].append(v)
        if not z:
            return 0., 0., 0.
        z = np.asarray(z)
        return float((z ** 2).mean()), float(z.mean()), float(max(np.mean(v) for v in per.values()))

    result = dict(data=a.data, fit_runs=a.fit_runs, test_runs=a.test_runs, verified_identical=True, prefixes=PREFIXES, auroc={})
    names = ('oracle_sq', 'oracle_signed', 'oracle_max_step')
    for n in PREFIXES:
        sn = np.array([scores(r, n) for r in clean]); sp = np.array([scores(r, n) for r in faulty])
        for j, name in enumerate(names):
            row = dict(all=auroc(sn[:, j], sp[:, j]))
            for k, kname in ((1, 'wear_and_tear'), (2, 'retry_delay')):
                row[kname] = auroc(sn[:, j], sp[kinds == k, j])
            result['auroc'].setdefault(name, {})[n] = row
            print(name, n, {k: round(v, 3) for k, v in row.items()}, flush=True)
    if a.out:
        out = ROOT / 'experiments/results/fas' / f'{a.out}.json'
        if out.exists():
            raise ValueError('Unique unused tag required')
        out.write_text(json.dumps(result, indent=1) + '\n')


if __name__ == '__main__':
    main()
