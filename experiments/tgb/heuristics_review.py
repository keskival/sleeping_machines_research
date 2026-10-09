#!/usr/bin/env python3
"""B5 protocol check on tgbl-review-v2 (AWS): official py-tgb loader, v2 negatives (100 per positive) and Evaluator end to
end with training-free time-decayed heuristics, sparse state (352K nodes): VALIDATION only; test stays sealed.

Scores for query (s, t) and candidate c from edges strictly before the current batch (batch 200, TGB convention):
  edgebank_inf: (s, c) ever observed;  pop_tau: decayed in-degree of c;  local_tau: decayed count of (s, c);
  local+pop combinations. Pickled negatives load through the allow-list unpickler (safe_tgb).
"""
import argparse, itertools, json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / '.cache/pylib_tgb')); sys.path.insert(0, str(Path(__file__).resolve().parent))
import safe_tgb  # noqa: E402

DAY = 86400.0
TAUS = (DAY, 7 * DAY, 30 * DAY, 365 * DAY)
ALPHAS = (1e-3, 1e-2, 0.1, 1.0)


class State:
    def __init__(self, n):
        self.P = np.zeros((len(TAUS), n)); self.plast = np.zeros(n)
        self.pair = {}                                      # (s, c) -> [A_taus..., last]
        self.T = np.array(TAUS)

    def add(self, s, c, t):
        for i in range(len(s)):
            si, ci, ti = int(s[i]), int(c[i]), float(t[i])
            e = self.pair.get((si, ci))
            if e is None:
                e = np.zeros(len(TAUS) + 1); self.pair[(si, ci)] = e
            e[:-1] = e[:-1] * np.exp(-(ti - e[-1]) / self.T) + 1.0; e[-1] = ti
            self.P[:, ci] = self.P[:, ci] * np.exp(-(ti - self.plast[ci]) / self.T) + 1.0; self.plast[ci] = ti

    def scores(self, s, cand, t):
        n = len(cand); loc = np.zeros((len(TAUS), n)); seen = np.zeros(n)
        for j, c in enumerate(cand):
            e = self.pair.get((s, int(c)))
            if e is not None:
                seen[j] = 1.0; loc[:, j] = e[:-1] * np.exp(-(t - e[-1]) / self.T)
        glo = self.P[:, cand] * np.exp(-(t - self.plast[cand])[None] / self.T[:, None])
        out = {'edgebank_inf': seen}
        for i in range(len(TAUS)):
            out[f'pop_{int(TAUS[i] / DAY)}d'] = glo[i]
            out[f'local_{int(TAUS[i] / DAY)}d'] = loc[i]
        for i, j, a in itertools.product(range(len(TAUS)), range(len(TAUS)), ALPHAS):
            out[f'lg_{int(TAUS[i] / DAY)}d_{int(TAUS[j] / DAY)}d_a{a}'] = loc[i] + a * glo[j] / max(glo[j].max(), 1e-12)
        return out


def rr(v):
    return 1.0 / (0.5 * ((v[1:] > v[0]).sum() + (v[1:] >= v[0]).sum()) + 1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True)
    ap.add_argument('--root', default='../../../data/tgb_aws/root')          # py-tgb prefixes its package directory
    ap.add_argument('--max-queries', type=int, default=0)
    a = ap.parse_args(); t0 = time.time(); safe_tgb.patch_tgb()
    from tgb.linkproppred.dataset import LinkPropPredDataset
    from tgb.linkproppred.evaluate import Evaluator
    ds = LinkPropPredDataset(name='tgbl-review', root=a.root, preprocess=True, download=False); ev = Evaluator(name='tgbl-review')
    d = ds.full_data; ds.load_val_ns(); n = int(max(d['sources'].max(), d['destinations'].max())) + 1
    st = State(n); tr = ds.train_mask
    st.add(d['sources'][tr], d['destinations'][tr], d['timestamps'][tr].astype(float))
    print(f'train state built {time.time() - t0:.0f}s, pairs {len(st.pair)}', flush=True)
    vm = ds.val_mask; src, dst, rt = d['sources'][vm], d['destinations'][vm], d['timestamps'][vm]
    if a.max_queries:
        src, dst, rt = src[:a.max_queries], dst[:a.max_queries], rt[:a.max_queries]
    mrr, checked, nneg = {}, 0, []
    for b in range(0, len(src), 200):
        s, c, r = src[b:b + 200], dst[b:b + 200], rt[b:b + 200]; t = r.astype(float)
        negs = ds.negative_sampler.query_batch(s, c, r, split_mode='val')
        for q in range(len(s)):
            ng = np.asarray(negs[q]); nneg.append(len(ng)); cand = np.concatenate([[c[q]], ng]).astype(np.int64)
            for k, v in st.scores(int(s[q]), cand, t[q]).items():
                x = rr(v); mrr.setdefault(k, []).append(x)
                if checked < 50:
                    o = ev.eval({'y_pred_pos': v[:1], 'y_pred_neg': v[None, 1:], 'eval_metric': ['mrr']})['mrr']
                    assert abs(float(o) - x) < 1e-6, (k, o, x)
            checked += 1
        st.add(s, c, t)
        if b % 100000 == 0:
            print(f'{b} queries {time.time() - t0:.0f}s', flush=True)
    val = {k: float(np.mean(v)) for k, v in mrr.items()}
    top = sorted(val.items(), key=lambda kv: -kv[1])[:10]
    out = dict(status='completed', tag=a.tag, battle='B5', dataset='tgbl-review (v2 negatives)', split='validation only',
               protocol='official py-tgb loader, negatives and Evaluator (checked on the first 50 queries); MRR',
               negatives_per_query=dict(min=int(min(nneg)), max=int(max(nneg))), queries=len(nneg), val_mrr=val,
               top10=top, wall_s=time.time() - t0)
    od = ROOT / 'experiments/results/tgb'; od.mkdir(parents=True, exist_ok=True)
    (od / f'{a.tag}.json').write_text(json.dumps(out, indent=1) + '\n')
    print('TOP', json.dumps(top), flush=True)


if __name__ == '__main__':
    main()
