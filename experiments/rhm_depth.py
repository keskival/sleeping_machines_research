#!/usr/bin/env python3
"""Depth trainability on a synthetic hierarchy with known depth (theory 173 section 14; founder, 17 October: depth is effective
learning over compositional depth, and useful depth grows with the logarithm of the data).

Data: a random hierarchy model (after Cagnetta, Petrini, Wyart et al.). A root symbol (v values) expands by one of m random
production rules into s symbols of the next level, L times; the s^L leaves are tokens. Rules are unambiguous (every s-tuple
belongs to at most one parent), so each level is a compositional decomposition of the one above. Sequences are trees separated
by a BOS token; trees are independent.

Reference ladder (exact, inside recursion over the fixed tree shape): B_k = mean per-leaf loss of the Bayes predictor that sees
only the earlier leaves inside the leaf's own level-k block (s^k leaves). B_0 is the position marginal, B_L full Bayes, and
B_{k-1} - B_k is the information that needs composition depth k. A model's resolved depth is where its loss falls on this
ladder (k + (B_k - loss)/(B_k - B_{k+1}); negative = worse than the marginal).

Question: does a K-level residual-memory member (r1_rmr.RMR, levels = K; K = 1 is the CERL race alone) resolve deeper
composition than K = 1, and at which sample count? A knockout of each correction level (its ReZero gates set to 0) shows which
composition depths that level carries. Fixed optimisation budget per run, so the sample count is the only data variable."""
import argparse, hashlib, json, math, resource, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
ROOT = Path(__file__).resolve().parents[1]; sys.path[:0] = [str(ROOT/'experiments'), str(ROOT/'experiments/tpp')]
from r1_cerl import word_classes, param_groups  # noqa: E402
from r1_rmr import RMR  # noqa: E402


class Hierarchy:
    def __init__(self, v, m, s, L, seed=0):
        assert m*v <= v**s, 'unambiguous rules need m v <= v^s'
        rng = np.random.default_rng(seed); self.v, self.m, self.s, self.L, self.n = v, m, s, L, s**L; self.rules = []
        for _ in range(L):                                                                        # level l: rules[l][parent, r] = s child symbols; all m v tuples distinct
            flat = rng.choice(v**s, size=m*v, replace=False)
            self.rules.append(np.stack([(flat//v**(s - 1 - c)) % v for c in range(s)], -1).reshape(v, m, s))

    def sample(self, N, rng):
        sym = rng.integers(0, self.v, (N, 1))
        for l in range(self.L): sym = self.rules[l][sym, rng.integers(0, self.m, sym.shape)].reshape(N, -1)
        return sym                                                                                # [N, n] leaves, node j's children at j s + c

    def log_prob(self, X, observed):
        """log P(leaves where observed [n] bool) for trees X [N, n]: inside recursion, unobserved leaves sum to 1."""
        N = X.shape[0]; beta = np.ones((N, self.n, self.v)); oh = np.eye(self.v)[X]; beta[:, observed] = oh[:, observed]
        for l in range(self.L - 1, -1, -1):
            R = self.rules[l]; nodes = beta.shape[1]//self.s; ch = beta.reshape(N, nodes, self.s, self.v); p = np.ones((N, nodes, self.v, self.m))
            for c in range(self.s): p = p*ch[:, :, c, :][:, :, R[:, :, c]]
            beta = p.mean(-1)
        return np.log(beta[:, 0].mean(-1))

    def bayes_nll(self, X):
        lp = [self.log_prob(X, np.arange(self.n) < t) for t in range(self.n + 1)]; return np.stack([lp[t] - lp[t + 1] for t in range(self.n)], 1)

    def marginal_nll(self, X):
        return np.stack([-self.log_prob(X, np.arange(self.n) == t) for t in range(self.n)], 1)

    def ladder(self, X):
        """[B_0, ..., B_L]: mean per-leaf loss with context truncated to the leaf's level-k block."""
        out = []
        for k in range(self.L + 1):
            tot = 0.0
            for t in range(self.n):
                a = (t//self.s**k)*self.s**k; r = np.arange(self.n); tot = tot + self.log_prob(X, (r >= a) & (r < t)) - self.log_prob(X, (r >= a) & (r <= t))
            out.append(float(np.mean(tot)/self.n))
        return out

    def depth_of(self, t):
        """Composition depth needed by leaf t: k + 1 for k trailing zeros of t in base s (t = 0: the tree start, prior only)."""
        if t == 0: return self.L + 1
        k = 0
        while t % self.s == 0: t //= self.s; k += 1
        return k + 1


def lanes(H, trees):
    """Trees [N, n] -> token lanes [N/T, T (n + 1) + 1]: BOS = v before each tree, a closing BOS as the last target."""
    N = trees.shape[0]; bos = np.full((N, 1), H.v); seq = np.concatenate([bos, trees], 1).reshape(-1)
    return seq


def contracts():
    res = {}; H = Hierarchy(3, 2, 2, 2, seed=1); rng = np.random.default_rng(0)
    # brute force: enumerate every tree (root, rule choices) with its probability; exact conditionals by counting
    from itertools import product
    probs = {}
    for root in range(3):
        for r0 in range(2):
            kids = H.rules[0][root, r0]
            for r1, r2 in product(range(2), range(2)):
                leaves = tuple(H.rules[1][kids[0], r1]) + tuple(H.rules[1][kids[1], r2]); probs[leaves] = probs.get(leaves, 0.0) + 1/(3*2*4)
    X = np.array(list(probs)); bn = H.bayes_nll(X); worst = 0.0
    for i, leaves in enumerate(map(tuple, X)):
        for t in range(4):
            num = sum(p for l_, p in probs.items() if l_[:t + 1] == leaves[:t + 1]); den = sum(p for l_, p in probs.items() if l_[:t] == leaves[:t])
            worst = max(worst, abs(-math.log(num/den) - bn[i, t]))
    res['bayes_equals_enumeration'] = worst
    H2 = Hierarchy(8, 4, 2, 4, seed=0); Xs = H2.sample(64, rng); tot = 0.0
    for t in (1, 5, 8, 15):
        base = H2.log_prob(Xs, np.arange(16) < t); s_ = 0.0
        for x in range(8):
            Xc = Xs.copy(); Xc[:, t] = x; s_ = s_ + np.exp(H2.log_prob(Xc, np.arange(16) < t + 1) - base)
        tot = max(tot, float(np.abs(s_ - 1).max()))
    res['bayes_normalisation_max_err'] = tot
    Xb = H2.sample(2048, rng); b, mg = H2.bayes_nll(Xb), H2.marginal_nll(Xb)
    res['bayes_below_marginal'] = float((mg - b)[:, 1:].mean()); res['bayes_equals_marginal_at_tree_start'] = float(np.abs(b[:, 0] - mg[:, 0]).max())
    lad = H2.ladder(Xb); res['ladder_ends'] = max(abs(lad[0] - float(mg.mean())), abs(lad[-1] - float(b.mean()))); res['ladder_monotone'] = all(x_ >= y_ - 1e-12 for x_, y_ in zip(lad, lad[1:]))
    res['status'] = 'passed' if res['ladder_ends'] < 1e-9 and res['ladder_monotone'] and res['bayes_equals_enumeration'] < 1e-9 and res['bayes_normalisation_max_err'] < 1e-9 and res['bayes_below_marginal'] > 0 and res['bayes_equals_marginal_at_tree_start'] < 1e-9 else 'failed'
    return res


def evaluate(model, H, test, T, ref):
    """Per-leaf model NLL on test trees, grouped by composition depth, against Bayes and marginal."""
    model.eval(); n = H.n; seq = lanes(H, test); per = T*(n + 1); out = np.zeros_like(ref['bayes'])
    with torch.no_grad():
        for i in range(0, test.shape[0]//T):
            w = torch.from_numpy(np.concatenate([seq[i*per:(i + 1)*per], [H.v]]).astype(np.int64))[None]; nll = model.lane_nll(w[:, :-1], w[:, 1:])[0].numpy()
            out[i*T:(i + 1)*T] = nll.reshape(T, n + 1)[:, :n]                                    # target j (n + 1) + t is leaf t of tree j (the BOS target closes the previous tree)
    model.train(); return dict(leaf_nll=float(out.mean()), resolved_depth=resolved_depth(float(out.mean()), ref['ladder']))


def resolved_depth(loss, B):
    """Position of a loss on the ladder B_0 >= ... >= B_L (piecewise linear; below 0 = worse than the marginal; capped at L)."""
    if loss >= B[0]: return (B[0] - loss)/(B[0] - B[1])
    for k in range(len(B) - 1):
        if loss >= B[k + 1]: return k + (B[k] - loss)/(B[k] - B[k + 1])
    return float(len(B) - 1)


def run(H, K, n_train, a, test, ref, seed):
    rng = np.random.default_rng(seed + 1); torch.manual_seed(seed); train = H.sample(n_train, rng); T = a.trees_per_lane; V = H.v + 1
    counts = np.bincount(np.concatenate([train.reshape(-1), np.full(n_train, H.v)]), minlength=V).astype(float); cls, words = word_classes(counts, a.classes, s_max=V)
    cerl = dict(d=a.d, modes=a.modes, layers=a.layers, dk=a.dk, units=a.units, topk=4, explore=2, unit_init=0.02, balance=0.01)
    nat = dict(read_topk=8, read_explore=2, write_explore=2, fire=True, events=True) if a.native else {}   # the native member: sparse reads with probes, async firing, composing event spine, slot level 1
    if a.native: cerl.update(level1='slots', slots=256, slot_m=4, slot_explore=2, slot_pk=True, slot_read_topk=8)
    model = RMR(V, cls, words, cerl, levels=K, dk=a.dk, top_m=a.top_m, chunk=a.chunk, old_decay=0.01 if a.old_decay else 0.0, old_per_target=bool(a.old_per_target), write_race=bool(a.write_race), **nat)
    if a.native:
        for t_ in model.base.tm: t_.CHUNK = min(t_.CHUNK, 64)
    opt = torch.optim.AdamW(param_groups(model, 0.01, True), lr=a.lr, weight_decay=0.01); sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05)
    per = T*(H.n + 1); t0 = time.time(); losses = []
    for step in range(a.steps):                                                                   # epochs over the fixed training trees, a fresh order each pass
        idx = rng.choice(n_train, size=a.batch*T, replace=n_train < a.batch*T); seq = lanes(H, train[idx]).reshape(a.batch, per)
        w = torch.from_numpy(np.concatenate([seq, np.full((a.batch, 1), H.v)], 1).astype(np.int64)); nll = model.lane_nll(w[:, :-1], w[:, 1:])
        obj = nll.mean() + (model.aux.mean() if model.aux is not None else 0.0); opt.zero_grad(); obj.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        losses.append(float(nll.mean()))
    res = dict(K=K, n_train=n_train, train_wall_s=time.time() - t0, train_loss_last=float(np.mean(losses[-100:])), test=evaluate(model, H, test, T, ref))
    knock = {}
    for j, lev in enumerate(model.levels):                                                       # knockout: level j + 2's ReZero gates to 0 (exactly removes its corrections)
        sv = (lev.alpha_new.data.clone(), lev.alpha_old.data.clone()); lev.alpha_new.data.zero_(); lev.alpha_old.data.zero_()
        knock[str(j + 2)] = evaluate(model, H, test, T, ref); lev.alpha_new.data.copy_(sv[0]); lev.alpha_old.data.copy_(sv[1])
    res['knockout'] = knock; return res


def main():
    ap = argparse.ArgumentParser(); A = ap.add_argument
    A('--mode', choices=('run', 'contract'), default='run'); A('--tag', default=''); A('--seed', type=int, default=0)
    A('--v', type=int, default=16); A('--m', type=int, default=4); A('--s', type=int, default=2); A('--L', type=int, default=4)
    A('--Ks', default='1,2,4'); A('--Ns', default='1024,4096,16384,65536'); A('--steps', type=int, default=3000); A('--batch', type=int, default=8); A('--trees-per-lane', type=int, default=8)
    A('--d', type=int, default=64); A('--modes', type=int, default=16); A('--layers', type=int, default=1); A('--dk', type=int, default=32); A('--units', type=int, default=128); A('--classes', type=int, default=3)
    A('--top-m', type=int, default=2); A('--chunk', type=int, default=32); A('--old-decay', type=int, default=0); A('--old-per-target', type=int, default=1); A('--write-race', type=int, default=1)
    A('--native', type=int, default=1); A('--lr', type=float, default=3e-3); A('--n-test', type=int, default=2048)
    a = ap.parse_args(); torch.set_num_threads(1)
    src = {n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ('experiments/rhm_depth.py', 'experiments/r1_rmr.py', 'experiments/r1_cerl.py', 'experiments/r1_deep_keyed_lm.py')}
    if a.mode == 'contract':
        res = dict(battle='R1', stage='synthetic hierarchy reference contracts', **contracts(), source_sha256=src); print(json.dumps(res))
        if a.tag: (ROOT/f'experiments/results/diagnostics/{a.tag}.json').open('x').write(json.dumps(res, indent=1) + '\n')
        return
    out = ROOT/f'experiments/results/diagnostics/{a.tag}.json'; assert a.tag and not out.exists()
    H = Hierarchy(a.v, a.m, a.s, a.L, seed=a.seed); test = H.sample(a.n_test, np.random.default_rng(10_000 + a.seed)); ref = dict(bayes=H.bayes_nll(test), ladder=H.ladder(test))
    runs = []
    for N in [int(z) for z in a.Ns.split(',')]:
        for K in [int(z) for z in a.Ks.split(',')]:
            runs.append(run(H, K, N, a, test, ref, a.seed)); r_ = runs[-1]
            print(json.dumps(dict(K=K, n_train=N, leaf_nll=r_['test']['leaf_nll'], ladder=[round(b_, 4) for b_ in ref['ladder']],
                                  resolved_depth=r_['test']['resolved_depth'], knockout_depth={j: k_['resolved_depth'] for j, k_ in r_['knockout'].items()}, wall_s=round(r_['train_wall_s']))), flush=True)
    res = dict(status='completed', battle='R1', stage='depth trainability on a synthetic hierarchy (theory 173 section 14)', tag=a.tag, args=vars(a),
               hierarchy=dict(v=a.v, m=a.m, s=a.s, L=a.L, leaves=H.n, ladder=ref['ladder'], ladder_gains=[ref['ladder'][k] - ref['ladder'][k + 1] for k in range(a.L)]), runs=runs,
               max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, source_sha256=src)
    out.write_text(json.dumps(res, indent=1) + '\n')


if __name__ == '__main__':
    main()
