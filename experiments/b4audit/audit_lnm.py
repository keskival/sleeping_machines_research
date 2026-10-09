#!/usr/bin/env python3
"""B4 audit (user exception, 9 Oct 2026): does the published log-normal-mixture model class exploit the timestamp grid?

Trains the reference authors' own GRU + log-normal-mixture model with their code (github.com/tanguybosser/ntpp-tmlr2023,
commit 54c15fd; the README's published LNM configuration, no tuning) on one dataset split, then scores the SAME
checkpoint with their own `evaluate()` on (a) the recorded test file and (b) a twin test file whose gaps are dequantized
uniformly inside the recording cell (zero gaps uniformly on [0, cell/2]); training and validation data are identical.
A model that does not exploit the grid scores about the same on both; a lattice-spiking one collapses on (b).
Only Hawkes simulation (tick) and version bookkeeping (git, torchvision) are shimmed (experiments/b4audit/shims).
Run under run_safe.
"""
import argparse, hashlib, json, os, shutil, sys, time
from argparse import Namespace
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / 'data/ntpp_bosser/code/repo'
SRC = ROOT / 'data/ntpp_bosser/extracted/processed/data'
sys.path[:0] = [str(ROOT / 'experiments/b4audit/shims'), str(ROOT / '.cache/pylib_ntpp'), str(REPO)]
sys.path.insert(0, str(ROOT / 'experiments/tpp'))
from race_tpp_b4 import recording_cell  # noqa: E402  (the same divisor rule as B4)


def build_root(ds, split, seed, out):
    """audit root with <ds> (symlink to the original) and <ds>_dq (identical except the dequantized test file)."""
    out.mkdir(parents=True, exist_ok=True)
    o = out / ds
    if not o.exists():
        o.symlink_to(SRC / ds)
    sd = SRC / ds / f'split_{split}'; dd = out / f'{ds}_dq' / f'split_{split}'; dd.mkdir(parents=True, exist_ok=True)
    for f in ('train.json', 'val.json', 'args.json'):
        shutil.copyfile(sd / f, dd / f)
    tr = json.loads((sd / 'train.json').read_text())
    g = np.concatenate([np.diff([e['time'] for e in s]) for s in tr]); cell = recording_cell(g[g > 0])
    rng = np.random.default_rng(seed); te = json.loads((sd / 'test.json').read_text()); out_seqs = []
    for s in te:
        t = np.array([e['time'] for e in s], float); gaps = np.diff(np.concatenate([[0.0], t]))
        u = rng.random(len(gaps))
        dq = np.where(gaps > 0, gaps + (u - 0.5) * cell, u * 0.5 * cell) if cell > 0 else gaps
        t2 = np.cumsum(np.maximum(dq, 1e-12))
        out_seqs.append([dict(time=float(a), labels=e['labels']) for a, e in zip(t2, s)])
    (dd / 'test.json').write_text(json.dumps(out_seqs))
    return cell


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--dataset', required=True); ap.add_argument('--split', type=int, default=0)
    ap.add_argument('--tag', required=True); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--max-epochs', type=int, default=1000)
    a = ap.parse_args(); t0 = time.time()
    aroot = ROOT / 'data/ntpp_bosser/audit_root'; cell = build_root(a.dataset, a.split, a.seed, aroot)
    os.chdir(REPO)
    from tpp.utils.cli import parse_args
    import scripts.train as T
    import torch as th
    th.set_num_threads(1)
    ck = ROOT / f'experiments/results/b4audit/{a.tag}_ckpt'; ck.mkdir(parents=True, exist_ok=True)
    readme_cfg = ['--dataset', a.dataset, '--load-from-dir', os.path.relpath(aroot, REPO), '--save-check-dir', str(ck),
                  '--eval-metrics', 'True', '--include-poisson', 'False', '--patience', '100', '--batch-size', '8',
                  '--split', str(a.split), '--encoder', 'gru', '--encoder-encoding', 'temporal_with_labels',
                  '--encoder-emb-dim', '8', '--encoder-units-rnn', '32', '--encoder-layers-rnn', '1', '--encoder-units-mlp', '16',
                  '--encoder-activation-mlp', 'relu', '--decoder', 'log-normal-mixture', '--decoder-units-mlp', '16',
                  '--decoder-units-mlp', '16', '--decoder-n-mixture', '32', '--seed', str(a.seed),
                  '--train-epochs', str(a.max_epochs), '--disable-cuda']
    sys.argv = ['train.py'] + readme_cfg
    args = parse_args()

    def with_json(args, ds):
        d = vars(args).copy(); d['dataset'] = ds
        js = json.loads((aroot / ds / f'split_{a.split}' / 'args.json').read_text()); d.update(js)
        n = Namespace(**d); n.mu = np.array(n.mu, dtype=np.float32)
        n.alpha = np.array(n.alpha, dtype=np.float32).reshape(n.mu.shape * 2); n.beta = np.array(n.beta, dtype=np.float32).reshape(n.mu.shape * 2)
        n.device = th.device('cpu'); return n

    args = with_json(args, a.dataset); T.make_deterministic(seed=args.seed)
    ds = T.load_data(args=args); L = {k: T.get_loader(ds[k], args=args, shuffle=(k == 'train')) for k in ds}
    model = T.get_model(args)
    model, _, _ = T.train(model, args=args, loader=L['train'], val_loader=L['val'])
    raw = T.evaluate(model=model, args=args, loader=L['test'], test=True)
    args_dq = with_json(args, f'{a.dataset}_dq'); ds_dq = T.load_data(args=args_dq)
    dq = T.evaluate(model=model, args=args_dq, loader=T.get_loader(ds_dq['test'], args=args_dq, shuffle=False), test=True)
    keep = ('loss', 'log ground density', 'log mark density', 'window integral')
    res = dict(status='completed', battle='B4-audit', tag=a.tag, dataset=a.dataset, split=a.split, recording_cell=cell,
               reference='github.com/tanguybosser/ntpp-tmlr2023@54c15fd, README LNM configuration (GRU, 32-component log-normal mixture)',
               parameters=int(T.count_parameters(model)), test_recorded={k: raw[k] for k in keep},
               test_dequantized={k: dq[k] for k in keep}, wall_s=time.time() - t0,
               source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in (Path(__file__).resolve(), REPO / 'scripts/train.py',
                                        REPO / 'tpp/models/decoders/log_normal_mixture.py')})
    od = ROOT / 'experiments/results/b4audit'; (od / f'{a.tag}.json').write_text(json.dumps(res, indent=1) + '\n')
    print('RESULT', json.dumps({k: res[k] for k in ('test_recorded', 'test_dequantized', 'parameters', 'wall_s')}), flush=True)


if __name__ == '__main__':
    main()
