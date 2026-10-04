"""Seeded FAS-Simulator challenge data (experiments/FAS_BENCHMARK.md).

Wraps the unmodified simulator (experiments/vendor/fas_simulator_3839b10; Keski-Valkama 2017, Procedia Computer Science
119) exactly as its generate_datasets.py does: one production run of 30 interleaved assembly items with the 10-second
clock, either clean or with one fault added at time 0 (wear and tear on a random conveyor, or retry delay on a random bowl
feeder, chosen uniformly).  Differences from the original script: every run is seeded (Python random and NumPy from
--seed-base + run index, disjoint per split), and timestamps are kept (the original NumPy export drops them).  Event ids
follow the original converter: clock events, then production-line events, in declaration order.

Output: experiments/data/fas/<name>/<split>.npz with ids (uint8), times_ms (int64), offsets (int64, run boundaries),
fault (0 clean, 1 wear and tear, 2 retry delay), target (faulty module name or ''), seed; plus manifest.json.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
VENDOR = ROOT / 'experiments/vendor/fas_simulator_3839b10'
sys.path.insert(0, str(ROOT / 'experiments/vendor/simpy_pkg')); sys.path.insert(0, str(VENDOR))
import simpy  # noqa: E402
from modules.components.clock import Clock  # noqa: E402
from modules.faults.retry_delay import RetryDelay  # noqa: E402
from modules.faults.wear_and_tear import WearAndTear  # noqa: E402
from modules.process.fas_instance import FASInstance  # noqa: E402
from modules.process.production_line import ProductionLine  # noqa: E402

EVENTS = Clock(None, None, False).get_events() + ProductionLine(None, None, False).get_events()
EVENT_ID = {e: i for i, e in enumerate(EVENTS)}
SPLITS = dict(train_clean=(0, 0), val_clean=(1, 0), val_faulty=(2, 1), test_clean=(3, 0), test_faulty=(4, 1))


def generate_run(seed, faulty, items=30):
    random.seed(seed); np.random.seed(seed % 2**32)
    env = simpy.Environment()

    class Log:                      # the simulator's Logger interface, keeping (time, type) without JSON round trips
        def __init__(self):
            self.lines = []

        def addMessage(self, msgtype, metadata=''):
            self.lines.append((env.now, msgtype))

    logger = Log()
    line = ProductionLine(env, logger, False)
    Clock(env, logger, False).spawn()
    kind, target = 0, ''
    if faulty:
        kind = random.choice([1, 2])

        def add_fault():
            nonlocal target
            yield env.timeout(0)
            if kind == 1:
                module = random.sample(line.conveyors, 1)[0]; module.add_fault(WearAndTear(env, module, False))
            else:
                module = random.sample(line.bowl_feeders, 1)[0]; module.add_fault(RetryDelay(env, module, False))
            target = module.name
        env.process(add_fault())
    last = None
    for _ in range(items):
        last = FASInstance(env, line, logger).spawn()
    env.run(last)
    times = np.array([t for t, _ in logger.lines], dtype=np.int64)
    ids = np.array([EVENT_ID[m] for _, m in logger.lines], dtype=np.uint8)
    return ids, times, kind, target


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name', required=True); p.add_argument('--seed-base', type=int, default=20261004)
    p.add_argument('--train', type=int, default=10000); p.add_argument('--val', type=int, default=1000)
    p.add_argument('--test', type=int, default=2000); p.add_argument('--items', type=int, default=30)
    a = p.parse_args()
    out = ROOT / 'experiments/data/fas' / a.name
    if out.exists():
        raise ValueError('Unique unused dataset name required')
    out.mkdir(parents=True)
    counts = dict(train_clean=a.train, val_clean=a.val, val_faulty=a.val, test_clean=a.test, test_faulty=a.test)
    manifest = dict(name=a.name, args=vars(a), events=EVENTS, simulator='keskival/FAS-Simulator@3839b10 (vendored, unmodified)',
                    seeds='seed = seed_base + 10_000_000 * split_index + run', splits={}, started=time.time())
    for split, (index, faulty) in SPLITS.items():
        ids, times, offsets, kinds, targets, seeds = [], [], [0], [], [], []
        t0 = time.perf_counter()
        for run in range(counts[split]):
            seed = a.seed_base + 10_000_000 * index + run
            i, t, k, g = generate_run(seed, faulty, a.items)
            ids.append(i); times.append(t); offsets.append(offsets[-1] + len(i)); kinds.append(k); targets.append(g)
            seeds.append(seed)
        path = out / f'{split}.npz'
        np.savez_compressed(path, ids=np.concatenate(ids), times_ms=np.concatenate(times), offsets=np.array(offsets),
                            fault=np.array(kinds, dtype=np.int8), target=np.array(targets), seed=np.array(seeds))
        lengths = np.diff(offsets)
        manifest['splits'][split] = dict(runs=counts[split], events=int(offsets[-1]), mean_events=float(lengths.mean()),
                                         wall_s=time.perf_counter() - t0,
                                         sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        print(json.dumps({split: manifest['splits'][split]}), flush=True)
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')


if __name__ == '__main__':
    main()
