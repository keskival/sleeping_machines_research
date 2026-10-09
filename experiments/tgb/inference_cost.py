#!/usr/bin/env python3
"""B5 measured inference cost on one CPU thread (latency does not depend on the trained weight values, so the models are
timed at their trained architecture with fresh initialisation).

tgbl-wiki (race_link v4, d = 0 and d = 16): per-event streaming as evaluated — advance the causal state, featurise the
query against all 1,000 candidate destinations, score them with the readout, push the event. Timed over 5,000
validation events after a 1,000-event warm-up, starting from the state after the training stream.
tgbn-trade (race_affinity v1): per label time, featurise all labelled nodes and score all destinations; timed over the
validation label times. Reports microseconds per query (per node prediction for trade) and parameters.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / '.cache/pylib_tgb')); sys.path.insert(0, str(Path(__file__).resolve().parent))


def wiki(out):
    import race_link_v4 as v4
    from tgb.linkproppred.dataset import LinkPropPredDataset
    ds = LinkPropPredDataset(name='tgbl-wiki', root=str(ROOT / 'data/tgb'), preprocess=True); d = ds.full_data; ds.load_val_ns()
    negset = set()
    for v in ds.negative_sampler.eval_set['val'].values():
        negset.update(int(x) for x in v)
    nodes = np.unique(np.concatenate([d['sources'], d['destinations'], np.fromiter(negset, np.int64)]))
    dests = np.unique(np.concatenate([d['destinations'], np.fromiter(negset, np.int64)]))
    sidx = np.full(nodes.max() + 1, -1, np.int64); sidx[nodes] = np.arange(len(nodes))
    didx = np.full(nodes.max() + 1, -1, np.int64); didx[dests] = np.arange(len(dests))
    S, Dn, T = sidx[d['sources']], didx[d['destinations']], d['timestamps'].astype(float)
    tr = np.flatnonzero(ds.train_mask); va = np.flatnonzero(ds.val_mask)
    for d_id in (0, 16):
        mem = v4.Memory(len(nodes), len(dests)); mem.add(S[tr], Dn[tr], T[tr])
        nf = v4.Memory(1, 1).features(np.zeros(1, np.int64), np.zeros(1)).shape[-1]
        model = v4.Readout(nf, 64, len(nodes), len(dests), d_id).eval()
        with torch.no_grad():
            for phase, rng in (('warm', va[:1000]), ('time', va[1000:6000])):
                t0 = time.perf_counter()
                for k in rng:
                    mem.advance(T[k]); x = mem.features(S[k:k + 1], T[k:k + 1])
                    model(torch.from_numpy(x), torch.from_numpy(S[k:k + 1])); mem.push(S[k], Dn[k], T[k])
                el = time.perf_counter() - t0
        out[f'tgbl_wiki_v4_d{d_id}'] = dict(us_per_query=1e6 * el / 5000, queries_per_second=5000 / el, candidates=len(dests),
                                            parameters=sum(p.numel() for p in model.parameters()))
        print(out[f'tgbl_wiki_v4_d{d_id}'], flush=True)


def trade(out):
    import race_affinity as ra
    from tgb.nodeproppred.dataset import NodePropPredDataset
    ds = NodePropPredDataset(name='tgbn-trade', root=str(ROOT / 'data/tgb'), preprocess=True); d = ds.full_data
    per = ra.Periods(ds, np.arange(int(d['destinations'].max()) + 1, dtype=np.int64))
    fired = ra.fired_label_times(ds, [('train', ds.train_mask), ('val', ds.val_mask), ('test', ds.test_mask)])
    model = ra.Readout(29, 32).eval(); n = 0; t0 = time.perf_counter()
    with torch.no_grad():
        for ts in fired['val']:
            nodes = np.array(list(ds.label_dict[ts].keys()), dtype=np.int64); n += len(nodes)
            model(torch.from_numpy(per.features(ts, nodes)))
    el = time.perf_counter() - t0
    out['tgbn_trade_v1'] = dict(us_per_node_prediction=1e6 * el / n, predictions=n, destinations=ds.num_classes,
                                parameters=sum(p.numel() for p in model.parameters()))
    print(out['tgbn_trade_v1'], flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True); a = ap.parse_args()
    torch.set_num_threads(1); out = dict(battle='B5', tag=a.tag, threads=1, device='cpu')
    trade(out); wiki(out)
    out['status'] = 'completed'
    (ROOT / f'experiments/results/tgb/{a.tag}.json').write_text(json.dumps(out, indent=1) + '\n')


if __name__ == '__main__':
    main()
