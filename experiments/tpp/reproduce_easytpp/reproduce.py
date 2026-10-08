#!/usr/bin/env python3
"""Third-party reproduction of the five EasyTPP wins (race-of-clocks models, frozen drivers).

    python experiments/tpp/reproduce_easytpp/reproduce.py <taxi|taobao|stackoverflow|retweet|amazon|all> [--seeds 0 1 2 3 4]

For each dataset: checks the frozen driver's sha256, downloads the HuggingFace `easytpp/<dataset>` release (train/dev/test)
if absent and checks each file's sha256 (the files the original runs scored: identical test event counts), trains each
seed with the exact original arguments (selection on dev; test scored once per seed) on one CPU thread, and prints the
test log-likelihood per event against the original runs and the best published model. Needs Python >= 3.10, torch, numpy.
"""
import argparse, hashlib, json, statistics, subprocess, sys, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CFG = json.loads((Path(__file__).resolve().parent / 'config.json').read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def prepare(ds):
    c = CFG[ds]
    if sha(ROOT / c['driver']) != c['driver_sha256']:
        sys.exit(f"{c['driver']}: sha256 mismatch (expected the frozen driver {c['driver_sha256'][:12]}…)")
    d = ROOT / 'data/easytpp' / ds; d.mkdir(parents=True, exist_ok=True)
    for split, want in c['data_sha256'].items():
        f = d / f'{split}.json'
        if not f.exists():
            urllib.request.urlretrieve(f'https://huggingface.co/datasets/easytpp/{ds}/resolve/main/{split}.json', f)
        if sha(f) != want:
            sys.exit(f'{f}: sha256 mismatch with the release the original runs scored')


def run(ds, seeds, stamp):
    c = CFG[ds]; vals = []
    for s in seeds:
        tag = f'thirdparty_{ds}_s{s}_{stamp}'
        cmd = [sys.executable, str(ROOT / c['driver']), '--dataset', ds, '--seed', str(s), '--threads', '1', '--tag', tag] + c['args']
        print('$', ' '.join(cmd[1:]), flush=True)
        subprocess.run(cmd, check=True, cwd=ROOT)
        r = json.loads((ROOT / f'experiments/results/tpp/{tag}.json').read_text())
        vals.append(r['test']['ll']); print(f"{ds} seed {s}: TEST {r['test']['ll']:.4f} ({r['test']['events']} events)", flush=True)
    m = statistics.mean(vals); sd = statistics.stdev(vals) if len(vals) > 1 else float('nan')
    name, pub = c['published']
    print(f"\n{ds}: this run {m:.4f} ± {sd:.4f} (n={len(vals)}); original {c['ours_aws'][0]:.4f} ± {c['ours_aws'][1]:.4f}; "
          f"best published {pub} ({name}); scored test events {c['scored_test_events']}\n", flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('dataset'); ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    ap.add_argument('--dry-run', action='store_true', help='verify driver and data checksums only')
    a = ap.parse_args(); stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    sets = list(CFG) if a.dataset == 'all' else [a.dataset]
    for ds in sets:
        prepare(ds); print(f'{ds}: driver and data verified', flush=True)
        if not a.dry_run:
            run(ds, a.seeds, stamp)


if __name__ == '__main__':
    main()
