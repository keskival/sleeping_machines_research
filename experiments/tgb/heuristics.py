#!/usr/bin/env python3
"""B5 protocol check on TGB dynamic link prediction (tgbl-wiki-v2): official loader, negatives and evaluator end to end
with training-free time-decayed heuristics, before any model work.

Scores for a query (s, t) and candidate destination c, from edges observed strictly before the current batch:
  EdgeBank-inf: 1 if (s, c) was ever observed;
  LocalGlobal(tau_l, tau_g, alpha): A_tau_l[s, c] + alpha * P_tau_g[c] / max P, where A is the exponentially decayed count
  of past (s, c) edges (time constant tau_l) and P the decayed global in-degree of c.
State is updated with each batch's positives after the batch is scored (batch 200, TGB's convention; EdgeBank does the
same). The grid is selected on VALIDATION MRR; TEST is scored once with the selected configuration. Run under run_safe.
"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / '.cache/pylib_tgb'))                                  # py-tgb 2.3.0 (python -m pip --target)
TAUS = (3600.0, 86400.0, 7 * 86400.0, 30 * 86400.0)
ALPHAS = (0.0, 1e-3, 1e-2, 0.1, 1.0)


class State:
    def __init__(self, n_src, n_dst):
        self.A = np.zeros((len(TAUS), n_src, n_dst), np.float32); self.last = np.zeros((n_src, n_dst))
        self.seen = np.zeros((n_src, n_dst), bool)
        self.P = np.zeros((len(TAUS), n_dst)); self.plast = np.zeros(n_dst)

    def add(self, s, c, t):
        for i in range(len(s)):
            si, ci, ti = s[i], c[i], t[i]
            dec = np.exp(-(ti - self.last[si, ci]) / np.array(TAUS))
            self.A[:, si, ci] = self.A[:, si, ci] * dec + 1.0; self.last[si, ci] = ti; self.seen[si, ci] = True
            dec = np.exp(-(ti - self.plast[ci]) / np.array(TAUS))
            self.P[:, ci] = self.P[:, ci] * dec + 1.0; self.plast[ci] = ti

    def scores(self, s, cand, t):
        """All configurations for one query: dict name -> scores over cand."""
        dt_l = t - self.last[s, cand]; dt_g = t - self.plast[cand]; out = {'edgebank_inf': self.seen[s, cand].astype(float)}
        loc = [self.A[i, s, cand] * np.exp(-dt_l / tau) for i, tau in enumerate(TAUS)]
        glo = []
        for i, tau in enumerate(TAUS):
            g = self.P[i, cand] * np.exp(-dt_g / tau); glo.append(g / max(g.max(), 1e-12))
        for (i, j, a) in itertools.product(range(len(TAUS)), range(len(TAUS)), ALPHAS):
            if a == 0.0 and j > 0:
                continue
            out[f'lg_tl{int(TAUS[i])}_tg{int(TAUS[j])}_a{a}'] = loc[i] + a * glo[j]
        return out


def rr(v):
    """Reciprocal rank of v[0] among v[1:], TGB's tie convention (mean of optimistic and pessimistic rank)."""
    return 1.0 / (0.5 * ((v[1:] > v[0]).sum() + (v[1:] >= v[0]).sum()) + 1)


def run_split(ds, ev, st, mask, split, sidx, didx, names=None, bs=200, n_check=50):
    d = ds.full_data; src, dst, raw_t = d['sources'][mask], d['destinations'][mask], d['timestamps'][mask]
    ns = ds.negative_sampler; mrr = {}; n_neg = []; checked = 0
    for b in range(0, len(src), bs):
        s, c, rt = src[b:b + bs], dst[b:b + bs], raw_t[b:b + bs]; t = rt.astype(float)
        negs = ns.query_batch(s, c, rt, split_mode=split)
        for q in range(len(s)):
            ng = np.asarray(negs[q]); n_neg.append(len(ng))
            cand = didx[np.concatenate([[c[q]], ng])]
            sc = st.scores(sidx[s[q]], cand, t[q])
            for k, v in sc.items():
                if names is None or k in names:
                    r = rr(v); mrr.setdefault(k, []).append(r)
                    if checked < n_check:                                             # agree with the official Evaluator
                        o = ev.eval({'y_pred_pos': v[:1], 'y_pred_neg': v[None, 1:], 'eval_metric': ['mrr']})['mrr']
                        assert abs(float(o) - r) < 1e-6, (k, o, r)
            checked += 1
        st.add(sidx[s], didx[c], t)
    return {k: float(np.mean(v)) for k, v in mrr.items()}, dict(min=int(min(n_neg)), max=int(max(n_neg)), mean=float(np.mean(n_neg)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True); ap.add_argument('--root', default=str(ROOT / 'data/tgb'))
    a = ap.parse_args(); t0 = time.time()
    from tgb.linkproppred.dataset import LinkPropPredDataset
    from tgb.linkproppred.evaluate import Evaluator
    ds = LinkPropPredDataset(name='tgbl-wiki', root=a.root, preprocess=True); ev = Evaluator(name='tgbl-wiki')
    d = ds.full_data; ds.load_val_ns(); ds.load_test_ns()
    # index maps only (no labels): rows over all nodes, columns over every node that can be a candidate destination
    negset = set()
    for sp in ('val', 'test'):
        for v in ds.negative_sampler.eval_set[sp].values():
            negset.update(int(x) for x in v)
    nodes = np.unique(np.concatenate([d['sources'], d['destinations'], np.fromiter(negset, np.int64)]))
    sidx = np.full(nodes.max() + 1, -1, np.int64); sidx[nodes] = np.arange(len(nodes))
    dests = np.unique(np.concatenate([d['destinations'], np.fromiter(negset, np.int64)]))
    didx = np.full(nodes.max() + 1, -1, np.int64); didx[dests] = np.arange(len(dests))
    st = State(len(nodes), len(dests)); tr = ds.train_mask
    print('nodes', len(nodes), 'candidate destinations', len(dests), flush=True)
    st.add(sidx[d['sources'][tr]], didx[d['destinations'][tr]], d['timestamps'][tr].astype(float))
    print(f'train state built {time.time() - t0:.0f}s', flush=True)
    val, nneg = run_split(ds, ev, st, ds.val_mask, 'val', sidx, didx)
    best = max((k for k in val if k != 'edgebank_inf'), key=val.get)
    print('val edgebank_inf', val['edgebank_inf'], 'best', best, val[best], f'{time.time() - t0:.0f}s', flush=True)
    test, nneg_t = run_split(ds, ev, st, ds.test_mask, 'test', sidx, didx, names={'edgebank_inf', best})
    out = dict(status='completed', tag=a.tag, battle='B5', dataset='tgbl-wiki (v2 negatives)', protocol=
               'official py-tgb loader, negatives and Evaluator; MRR; selection on validation; test scored once',
               negatives_per_query=dict(val=nneg, test=nneg_t),
               val_mrr=val, selected=best, test_mrr=test, wall_s=time.time() - t0)
    (ROOT / f'experiments/results/tgb/{a.tag}.json').write_text(json.dumps(out, indent=1) + '\n')
    print('RESULT', json.dumps({k: out[k] for k in ('selected', 'test_mrr', 'negatives_per_query', 'wall_s')}))


if __name__ == '__main__':
    main()
