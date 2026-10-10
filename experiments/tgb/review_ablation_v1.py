#!/usr/bin/env python3
"""B5 tgbl-review development diagnosis (VALIDATION only, no training, no TEST).

The v3 full-development epoch scores val MRR 0.268, below the training-free 30-day popularity heuristic (0.340), and its
query profile is inverted: new-pair/known-source queries 0.247 vs new-source queries 0.475. Hypothesis: the 11.6M
source/destination identity parameters (es, ed, bd), trained one epoch, damage ranking for known sources. This driver
loads the saved checkpoint, rebuilds the identical causal state by replaying TRAIN (race_link_review_v2.CausalState,
v3's amortized sampler is irrelevant here: no negatives are sampled), and scores every official validation query with:
  full          the saved model (must reproduce the recorded 0.26783375086034844)
  mlp           feature network only (no identities)
  mlp_bd        feature network + destination bias
  ident         identity terms only (es.ed + bd)
  pop30, pop7   log1p decayed destination popularity (monotone in the heuristic's pop_30d / pop_7d)
  pop30_mlp     pop30 + feature network (residual-on-heuristic form)
Per query class (repeat pair / new pair of known source / new source). Official Evaluator agreement asserted.
"""
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import race_link_review as base  # noqa: E402
import race_link_review_v2 as v2  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True); ap.add_argument('--dataset', default='tgbl-review')
    ap.add_argument('--root', default='../../../data/tgb_aws/root')
    ap.add_argument('--checkpoint', required=True); ap.add_argument('--expect-full-mrr', type=float, default=None)
    ap.add_argument('--dim', type=int, default=16); ap.add_argument('--hidden', type=int, default=64)
    ap.add_argument('--max-val', type=int, default=0)
    a = ap.parse_args(); torch.set_num_threads(1); t0 = time.time()
    base.safe_tgb.patch_tgb()
    from tgb.linkproppred.dataset import LinkPropPredDataset
    from tgb.linkproppred.evaluate import Evaluator
    ds = LinkPropPredDataset(name=a.dataset, root=a.root, preprocess=True, download=False); ev = Evaluator(name=a.dataset)
    ds.load_val_ns(); d = ds.full_data; n = int(max(d['sources'].max(), d['destinations'].max())) + 1
    model = base.Model(n, a.dim, a.hidden); model.load_state_dict(torch.load(ROOT / a.checkpoint)); model.eval()
    st = v2.CausalState(n); tr = np.flatnonzero(ds.train_mask)
    st.history(d['sources'][tr], d['destinations'][tr], d['timestamps'][tr].astype(float)); replay_s = time.time() - t0
    mask = ds.val_mask; src = d['sources'][mask]; dst = d['destinations'][mask]; rt = d['timestamps'][mask]
    if a.max_val: src, dst, rt = src[:a.max_val], dst[:a.max_val], rt[:a.max_val]
    names = ('full', 'mlp', 'mlp_bd', 'ident', 'pop30', 'pop7', 'pop30_mlp')
    rr = {k: [] for k in names}; cls = []; checked = 0
    with torch.no_grad():
        for b in range(0, len(src), 200):
            s, c, r = src[b:b + 200], dst[b:b + 200], rt[b:b + 200]
            neg = ds.negative_sampler.query_batch(s, c, r, split_mode='val')
            cand = np.stack([np.concatenate(([c[q]], np.asarray(neg[q]))) for q in range(len(s))]).astype(np.int64)
            cand, feat = v2.causal_batch(st, s, c, r.astype(float), cand)
            S = torch.from_numpy(s.astype(np.int64)); C = torch.from_numpy(cand); F = torch.from_numpy(feat)
            mlp = model.mlp(model.norm(F)).squeeze(-1); ident = (model.es(S)[:, None] * model.ed(C)).sum(-1); bd = model.bd(C).squeeze(-1)
            sc = dict(full=model(S, C, F), mlp=mlp, mlp_bd=mlp + bd, ident=ident + bd, pop30=F[..., 9], pop7=F[..., 8],
                      pop30_mlp=F[..., 9] + mlp)
            for q in range(len(s)):
                cls.append('repeat_pair' if feat[q, 0, 5] > 0 else ('new_pair_known_source' if feat[q, 0, 14] > 0 else 'new_source'))
                for k in names:
                    rr[k].append(base.rr(sc[k][q].numpy()))
                if checked < 50:
                    off = ev.eval(dict(y_pred_pos=sc['full'][q, :1].numpy(), y_pred_neg=sc['full'][q, None, 1:].numpy(), eval_metric=['mrr']))['mrr']
                    assert abs(float(off) - rr['full'][-1]) < 1e-6; checked += 1
    cls = np.array(cls)
    out = {k: dict(mrr=float(np.mean(v)), **{c: float(np.mean(np.array(v)[cls == c])) for c in np.unique(cls)}) for k, v in rr.items()}
    res = dict(status='completed', battle='B5 tgbl-review development (diagnosis)', tag=a.tag, args=vars(a),
               checkpoint_sha256=hashlib.sha256((ROOT / a.checkpoint).read_bytes()).hexdigest(),
               queries=len(cls), class_counts={c: int((cls == c).sum()) for c in np.unique(cls)}, scorers=out,
               replay_wall_s=replay_s, wall_s=time.time() - t0, scope='VALIDATION only; no training; no TEST')
    if a.expect_full_mrr is not None:
        res['full_reproduction'] = dict(expected=a.expect_full_mrr, got=out['full']['mrr'], abs_diff=abs(out['full']['mrr'] - a.expect_full_mrr))
        if res['full_reproduction']['abs_diff'] > 1e-9:
            res['status'] = 'failed'
    res['source_sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in (Path(__file__).resolve(), HERE / 'race_link_review.py', HERE / 'race_link_review_v2.py', HERE / 'safe_tgb.py')}
    od = ROOT / 'experiments/results/tgb'; (od / f'{a.tag}.json').write_text(json.dumps(res, indent=1) + '\n')
    print(json.dumps({k: v['mrr'] for k, v in out.items()}), res['status'], flush=True)
    if res['status'] != 'completed':
        sys.exit(1)


if __name__ == '__main__':
    main()
