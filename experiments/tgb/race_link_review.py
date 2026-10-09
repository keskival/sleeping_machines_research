#!/usr/bin/env python3
"""B5 native model for large TGB link datasets (tgbl-review-v2 first): a race among candidate destinations driven by
sparse addressed temporal memory. AWS lineage of race_link v1 (curie), rebuilt for 352K nodes.

State (exact, addressed, updated event by event; only touched entries exist):
  pair slot (s, c):   decayed counts at TAUS, count, last time        -> keys = (s, c) addresses, values = clock sums
  destination c:      decayed in-degree at TAUS, count, last time     (popularity momentum across time scales)
  source s:           count, last time                                (activity clock)
  learned:            source / destination embeddings and a destination bias (transductive identity)
Score(s, c, t) = MLP(phi(s, c, t)) + <e_s, e_c> + b_c.  P(winner = c | candidates) = softmax over the candidate set: a race
of candidate clocks with hazards exp(score); every sampled loser receives credit. Training: time-ordered batches of 200,
features from state strictly before the batch, candidates = positive + K sampled competitors (half from destinations
already seen, half uniform over nodes: TGB's historical-plus-random scheme); the batch's positives then enter the state.
Evaluation: the official v2 negatives (100 per positive) and Evaluator (agreement asserted on the first 50 queries);
selection on VALIDATION MRR; TEST only with --score-test. Run under run_safe.
"""
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / '.cache/pylib_tgb')); sys.path.insert(0, str(Path(__file__).resolve().parent))
import safe_tgb  # noqa: E402

DAY = 86400.0
TAUS = np.array([DAY, 7 * DAY, 30 * DAY, 365 * DAY])
NF = 7 + 7 + 2


def slog(x):
    return np.sign(x) * np.log1p(np.abs(x))


class State:
    def __init__(self, n):
        self.n = n
        self.pop = np.zeros((n, len(TAUS))); self.dcnt = np.zeros(n); self.dlast = np.full(n, np.nan)
        self.scnt = np.zeros(n); self.slast = np.full(n, np.nan)
        self.pidx = {}; cap = 1 << 22
        self.pa = np.zeros((cap, len(TAUS))); self.pc = np.zeros(cap); self.pl = np.zeros(cap)
        self.seen_dst = np.zeros(n, bool); self.seen_list = []

    def add(self, s, c, t):
        for k in range(len(s)):
            u, p, tk = int(s[k]), int(c[k]), float(t[k])
            i = self.pidx.get((u, p))
            if i is None:
                i = len(self.pidx)
                if i >= len(self.pc):
                    self.pa = np.concatenate([self.pa, np.zeros_like(self.pa)])
                    self.pc = np.concatenate([self.pc, np.zeros_like(self.pc)]); self.pl = np.concatenate([self.pl, np.zeros_like(self.pl)])
                self.pidx[(u, p)] = i
            self.pa[i] = self.pa[i] * np.exp(-(tk - self.pl[i]) / TAUS) + 1.0; self.pc[i] += 1; self.pl[i] = tk
            dl = self.dlast[p]
            self.pop[p] = (self.pop[p] * np.exp(-(tk - dl) / TAUS) if not np.isnan(dl) else 0.0) + 1.0
            self.dcnt[p] += 1; self.dlast[p] = tk
            self.scnt[u] += 1; self.slast[u] = tk
            if not self.seen_dst[p]:
                self.seen_dst[p] = True; self.seen_list.append(p)

    def features(self, s, cand, t):
        """s [Q], cand [Q, C], t [Q] -> float32 [Q, C, NF]."""
        Q, C = cand.shape
        pair = np.zeros((Q, C, 7)); pair[..., 6] = slog(1e9)
        for q in range(Q):
            u = int(s[q])
            for j in range(C):
                i = self.pidx.get((u, int(cand[q, j])))
                if i is not None:
                    dt = t[q] - self.pl[i]
                    pair[q, j, :4] = np.log1p(self.pa[i] * np.exp(-dt / TAUS)); pair[q, j, 4] = np.log1p(self.pc[i])
                    pair[q, j, 5] = 1.0; pair[q, j, 6] = slog(dt)
        dl = self.dlast[cand]; ddt = t[:, None] - dl; never = np.isnan(dl)
        popd = np.where(never[..., None], 0.0, self.pop[cand] * np.exp(-np.where(never, 0.0, ddt)[..., None] / TAUS))
        dst = np.concatenate([np.log1p(popd), np.log1p(self.dcnt[cand])[..., None], slog(np.where(never, 1e9, ddt))[..., None],
                              (~never)[..., None].astype(float)], -1)[..., :7]
        sl = self.slast[s]; sdt = np.where(np.isnan(sl), 1e9, t - np.nan_to_num(sl))
        src = np.broadcast_to(np.stack([np.log1p(self.scnt[s]), slog(sdt)], -1)[:, None], (Q, C, 2))
        return np.concatenate([pair, dst, src], -1).astype(np.float32)

    def sample_negatives(self, k, rng, Q):
        hist = np.asarray(self.seen_list, np.int64)
        kh = k // 2 if len(hist) else 0
        h = hist[rng.integers(0, len(hist), (Q, kh))] if kh else np.zeros((Q, 0), np.int64)
        r = rng.integers(0, self.n, (Q, k - kh))
        return np.concatenate([h, r], 1)


class Model(nn.Module):
    def __init__(self, n, dim, hidden):
        super().__init__()
        self.norm = nn.LayerNorm(NF)
        self.mlp = nn.Sequential(nn.Linear(NF, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.es = nn.Embedding(n, dim); self.ed = nn.Embedding(n, dim); self.bd = nn.Embedding(n, 1)
        for e in (self.es, self.ed):
            nn.init.normal_(e.weight, std=0.01)
        nn.init.zeros_(self.bd.weight)

    def forward(self, s, cand, feat):
        x = self.mlp(self.norm(feat)).squeeze(-1)
        return x + (self.es(s)[:, None] * self.ed(cand)).sum(-1) + self.bd(cand).squeeze(-1)


def rr(v):
    return 1.0 / (0.5 * ((v[1:] > v[0]).sum() + (v[1:] >= v[0]).sum()) + 1)


def evaluate(ds, ev, model, st, mask, split, max_q=0):
    d = ds.full_data; src, dst, rt = d['sources'][mask], d['destinations'][mask], d['timestamps'][mask]
    if max_q:
        src, dst, rt = src[:max_q], dst[:max_q], rt[:max_q]
    model.eval(); out = []; checked = 0
    with torch.no_grad():
        for b in range(0, len(src), 200):
            s, c, r = src[b:b + 200], dst[b:b + 200], rt[b:b + 200]; t = r.astype(float)
            negs = ds.negative_sampler.query_batch(s, c, r, split_mode=split)
            cand = np.stack([np.concatenate([[c[q]], np.asarray(negs[q])]) for q in range(len(s))]).astype(np.int64)
            sc = model(torch.from_numpy(s.astype(np.int64)), torch.from_numpy(cand),
                       torch.from_numpy(st.features(s, cand, t))).numpy()
            for q in range(len(s)):
                x = rr(sc[q]); out.append(x)
                if checked < 50:
                    o = ev.eval({'y_pred_pos': sc[q, :1], 'y_pred_neg': sc[q, None, 1:], 'eval_metric': ['mrr']})['mrr']
                    assert abs(float(o) - x) < 1e-6, (o, x); checked += 1
            st.add(s, c, t)
    return float(np.mean(out)), len(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True); ap.add_argument('--dataset', default='tgbl-review')
    ap.add_argument('--root', default='../../../data/tgb_aws/root'); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=6); ap.add_argument('--neg', type=int, default=20)
    ap.add_argument('--dim', type=int, default=16); ap.add_argument('--hidden', type=int, default=64)
    ap.add_argument('--lr', type=float, default=3e-3); ap.add_argument('--max-train', type=int, default=0)
    ap.add_argument('--max-val', type=int, default=0); ap.add_argument('--score-test', action='store_true')
    a = ap.parse_args(); t0 = time.time()
    torch.set_num_threads(1); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed); safe_tgb.patch_tgb()
    from tgb.linkproppred.dataset import LinkPropPredDataset
    from tgb.linkproppred.evaluate import Evaluator
    ds = LinkPropPredDataset(name=a.dataset, root=a.root, preprocess=True, download=False); ev = Evaluator(name=a.dataset)
    d = ds.full_data; ds.load_val_ns(); n = int(max(d['sources'].max(), d['destinations'].max())) + 1
    tr = np.where(ds.train_mask)[0]
    if a.max_train:
        tr = tr[:a.max_train]
    model = Model(n, a.dim, a.hidden); opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    od = ROOT / 'experiments/results/tgb'; od.mkdir(parents=True, exist_ok=True); ckpt = od / f'{a.tag}.pt'
    best, best_ep, hist = -1.0, -1, []
    for ep in range(a.epochs):
        st = State(n); model.train(); e0 = time.time(); tot = 0.0; nb = 0
        for b in range(0, len(tr), 200):
            ix = tr[b:b + 200]; s, c, t = d['sources'][ix], d['destinations'][ix], d['timestamps'][ix].astype(float)
            cand = np.concatenate([c[:, None], st.sample_negatives(a.neg, rng, len(s))], 1).astype(np.int64)
            sc = model(torch.from_numpy(s.astype(np.int64)), torch.from_numpy(cand), torch.from_numpy(st.features(s, cand, t)))
            loss = F.cross_entropy(sc, torch.zeros(len(s), dtype=torch.long))
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item(); nb += 1
            st.add(s, c, t)
        if a.max_train:                                                               # state must cover all train events
            rest = np.where(ds.train_mask)[0][a.max_train:]
            st.add(d['sources'][rest], d['destinations'][rest], d['timestamps'][rest].astype(float))
        vm, nq = evaluate(ds, ev, model, st, ds.val_mask, 'val', a.max_val)
        hist.append(dict(epoch=ep, train_loss=tot / max(nb, 1), val_mrr=vm, val_queries=nq, epoch_s=time.time() - e0))
        print(json.dumps(hist[-1]), flush=True)
        if vm > best:
            best, best_ep = vm, ep; torch.save(model.state_dict(), ckpt)
    res = dict(status='completed', battle='B5', tag=a.tag, dataset=a.dataset, args=vars(a), nodes=n,
               parameters=sum(p.numel() for p in model.parameters()), best_epoch=best_ep, val_mrr=best, history=hist,
               checkpoint=str(ckpt.relative_to(ROOT)), wall_s=time.time() - t0)
    if a.score_test:
        model.load_state_dict(torch.load(ckpt)); ds.load_test_ns()
        st = State(n); st.add(d['sources'][ds.train_mask], d['destinations'][ds.train_mask], d['timestamps'][ds.train_mask].astype(float))
        st.add(d['sources'][ds.val_mask], d['destinations'][ds.val_mask], d['timestamps'][ds.val_mask].astype(float))
        res['test_mrr'], _ = evaluate(ds, ev, model, st, ds.test_mask, 'test')
    res['source_sha256'] = {str(p.relative_to(ROOT)): __import__('hashlib').sha256(p.read_bytes()).hexdigest()
                            for p in (Path(__file__).resolve(), Path(__file__).resolve().parent / 'safe_tgb.py')}
    (od / f'{a.tag}.json').write_text(json.dumps(res, indent=1) + '\n')
    print('RESULT', json.dumps({k: res[k] for k in ('val_mrr', 'best_epoch', 'parameters', 'wall_s')}), flush=True)


if __name__ == '__main__':
    main()
