#!/usr/bin/env python3
"""One-command reproduction of the TGB tgbl-wiki result (B5). Verifies the driver and data checksums, runs seeds 0-2 of
the frozen configuration with test scoring through the official py-tgb loader and Evaluator, and compares with the
recorded results. About 50 minutes per seed on one CPU thread; needs `pip install py-tgb==2.3.0` (or the --target install
used here: python -m pip install --target .cache/pylib_tgb py-tgb==2.3.0).

    python experiments/tgb/reproduce_wiki/reproduce.py [--data-root DIR]
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[2]
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--data-root', default=str(ROOT / 'data/tgb')); a = ap.parse_args()
    cfg = json.loads((HERE / 'config.json').read_text())
    assert sha(ROOT / cfg['driver']) == cfg['driver_sha256'], 'driver changed since the sealed runs'
    print('driver checksum ok', flush=True)
    got = []
    for s in cfg['seeds']:
        tag = f'repro_wiki_s{s}'
        subprocess.run([sys.executable, str(ROOT / cfg['driver']), '--tag', tag, '--seed', str(s), *cfg['args'],
                        '--root', a.data_root], check=True)
        got.append(json.loads((ROOT / f'experiments/results/tgb/{tag}.json').read_text())['test_mrr'])
    sys.path.insert(0, str(ROOT / '.cache/pylib_tgb'))
    import tgb
    data_dir = next(Path(tgb.__file__).parent.rglob('tgbl-wiki_edgelist_v2.csv')).parent
    for name, h in cfg['data_sha256'].items():
        assert sha(data_dir / name) == h, f'data file differs: {name}'
    print('data checksums ok')
    ok = all(e is None or abs(g - e) <= cfg['tolerance'] for g, e in zip(got, cfg['expected_test_mrr']))
    mean = sum(got) / len(got)
    print(json.dumps(dict(test_mrr=got, mean=mean, expected=cfg['expected_test_mrr'], reproduced=ok,
                          above_tpnet_0_827=all(g > 0.827 for g in got))))


if __name__ == '__main__':
    main()
