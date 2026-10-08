#!/usr/bin/env python3
"""B3 Stage 4: score one trained native model on the sealed FAS v2 test, once, and record it in the ledger.

Protocol (FAS_V2_CONFIRMATORY_PROTOCOL.md, Stage 4): each trained model is scored on test exactly once; every scoring is
appended to results/fas/fas_v2_test_ledger.jsonl with tag, configuration hash, checkpoint hash and UTC time. This script
refuses to score a tag that is already in the ledger. It loads the model from a race_tpp_fas(_vN) result JSON, scores
test_clean and test_faulty with native.py's rules and prefixes, writes per-run scores (for the paired bootstrap) and the
AUROC table, and appends the ledger entry. Training stays in the validation-only drivers.
"""
import argparse
import datetime
import hashlib
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/fas')); sys.path.insert(0, str(ROOT / 'experiments/tpp'))
LEDGER = ROOT / 'experiments/results/fas/fas_v2_test_ledger.jsonl'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--result', required=True, help='validation result JSON of the trained model')
    ap.add_argument('--reason', default='Stage 4 confirmation (first and only scoring)')
    a = ap.parse_args()
    torch.set_default_dtype(torch.float64); torch.set_num_threads(1)
    res = json.loads((ROOT / a.result).read_text()); args = res['args']; tag = res['tag']
    if LEDGER.exists() and any(json.loads(l).get('tag') == tag for l in LEDGER.read_text().splitlines() if l.strip()):
        raise SystemExit(f'{tag} is already in the test ledger; a second scoring must be declared with a reason first')
    R = importlib.import_module(next(Path(k).stem for k in res['source_sha256'] if 'race_tpp_fas' in k))
    d = ROOT / 'experiments/data/fas' / args['data']
    train, _ = R.load(d / 'train_clean.npz', args['train_max_events']); train = train[:args['fit_runs']]
    gaps = np.concatenate([np.diff(ts) for _, ts in train[:500]]); pos = gaps[gaps > 0]
    scale = float(np.median(pos)); qs = np.log(np.quantile(pos, np.linspace(0.1, 0.9, args['n_lognormal']))).tolist()
    edges = R.cluster_windows(pos, args['n_window'], args['seed']) if args['n_window'] else None
    margs = (46, args['d'], args['modes'], args['layers'], args['n_exp'], args['n_lognormal'], args['dv'], args['dropout'],
             scale, qs, args['cell_s'], args['n_window'], edges, 0)
    kw = dict(dk=args['dk'], local=args['keyed'] == 2)
    if args.get('pred_window'):
        kw.update(pred_window=args['pred_window'], keyed=bool(args['keyed']))
    if args.get('consume'):
        kw['consume'] = True
    model = R.KeyedRaceFAS(*margs, **kw) if (args['keyed'] or args.get('pred_window')) else R.RaceTPP(*margs)
    ckpt = ROOT / res['checkpoint']; model.load_state_dict(torch.load(ckpt)); model.eval()
    te_c, _ = R.load(d / 'test_clean.npz', args['max_events']); te_f, te_k = R.load(d / 'test_faulty.npz', args['max_events'])
    eb = args.get('eval_batch', 8)
    PTc, PGc, _ = R.per_position(model, te_c, eb); PTf, PGf, _ = R.per_position(model, te_f, eb)
    sc_c = R.prefix_scores(PTc, PGc, [len(r[0]) for r in te_c]); sc_f = R.prefix_scores(PTf, PGf, [len(r[0]) for r in te_f])
    table = R.aurocs(sc_c, sc_f, np.asarray(te_k))
    out = ROOT / 'experiments/results/fas' / f'{tag}_TEST'
    np.savez_compressed(str(out) + '_scores.npz', **{f'clean_{r}': v for r, v in sc_c.items()},
                        **{f'faulty_{r}': v for r, v in sc_f.items()}, kinds=np.asarray(te_k))
    Path(str(out) + '.json').write_text(json.dumps(dict(tag=tag, test_auroc=table, prefixes=R.PREFIXES), indent=1,
                                                       default=float) + '\n')
    entry = dict(tag=tag, config_sha256=hashlib.sha256(json.dumps(args, sort_keys=True).encode()).hexdigest(),
                 checkpoint_sha256=sha(ckpt), driver=next(k for k in res['source_sha256'] if 'race_tpp_fas' in k),
                 driver_sha256=next(v for k, v in res['source_sha256'].items() if 'race_tpp_fas' in k),
                 utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'), reason=a.reason)
    with LEDGER.open('a') as f:
        f.write(json.dumps(entry) + '\n')
    print('TEST', json.dumps(dict(tag=tag, total=table.get(1024), type=table['rules']['type'].get(1024)), default=float))


if __name__ == '__main__':
    main()
