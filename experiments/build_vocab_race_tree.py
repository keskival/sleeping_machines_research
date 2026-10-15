#!/usr/bin/env python3
"""Build the race-tree over the GPT-2 vocabulary from TRAIN statistics only (no validation or test tokens).

Token vectors: positive PMI of each token with its left and right neighbours among the 2,048 most frequent TRAIN tokens,
reduced to 64 dimensions by truncated SVD; tokens never seen in the sample get small random vectors. Tree: recursive
capacity-balanced 16-way k-means (four levels, 16^4 = 65,536 >= 50,257 leaves), so siblings are distributionally similar
tokens and every internal node has at most 16 children. Output: experiments/results/token_language/<tag>.npz with
path[v] = (c1, c2, c3, c4) for every token v, and a JSON record."""
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT/'experiments'))
from r1_token_keyed_lm import load, V

B, DEPTH = 16, 4


def balanced_split(X, ids, cap, rng):
    """Split ids into <= B groups of at most cap members by k-means centroids and greedy capacity-constrained assignment."""
    k = min(B, int(np.ceil(len(ids)/cap)) if cap > 0 else B, len(ids)); k = max(k, 1)
    if k == 1: return [ids]
    km = KMeans(n_clusters=k, n_init=1, random_state=int(rng.integers(1 << 30)), max_iter=50).fit(X[ids])
    d = ((X[ids][:, None, :] - km.cluster_centers_[None])**2).sum(-1); order = np.argsort(d.min(1) - np.sort(d, 1)[:, min(1, k - 1)])
    groups = [[] for _ in range(k)]
    for i in order:
        for c in np.argsort(d[i]):
            if len(groups[c]) < cap: groups[c].append(ids[i]); break
    return [np.array(g, dtype=np.int64) for g in groups if g]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True); ap.add_argument('--tokens', type=int, default=20_000_000); ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args(); t0 = time.time(); rng = np.random.default_rng(a.seed)
    out = ROOT/f'experiments/results/token_language/{a.tag}'; assert not out.with_suffix('.json').exists()
    x = np.asarray(load(ROOT/'data/fineweb_gpt2/fineweb_train_000001.bin')[:a.tokens]).astype(np.int64)
    freq = np.bincount(x, minlength=V); top = np.argsort(-freq)[:2048]; rank = np.full(V, -1); rank[top] = np.arange(2048)
    rows, cols = [], []
    for off, base in ((1, 0), (-1, 2048)):                                       # right and left neighbour contexts
        cur, nb = (x[:-1], x[1:]) if off == 1 else (x[1:], x[:-1]); keep = rank[nb] >= 0
        rows.append(cur[keep]); cols.append(rank[nb[keep]] + base)
    M = sp.coo_matrix((np.ones(sum(len(r) for r in rows)), (np.concatenate(rows), np.concatenate(cols))), shape=(V, 4096)).tocsr()
    tot = M.sum(); rs = np.asarray(M.sum(1)).ravel(); cs = np.asarray(M.sum(0)).ravel(); M = M.tocoo()
    pmi = np.log(M.data*tot/(rs[M.row]*cs[M.col])); keep = pmi > 0
    P = sp.coo_matrix((pmi[keep], (M.row[keep], M.col[keep])), shape=(V, 4096)).tocsr()
    X = TruncatedSVD(64, random_state=a.seed).fit_transform(P); X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    unseen = freq == 0; X[unseen] = rng.normal(scale=0.01, size=(int(unseen.sum()), 64))
    path = np.zeros((V, DEPTH), dtype=np.int64)
    stack = [(np.arange(V), 0, [])]
    while stack:
        ids, level, prefix = stack.pop()
        if level == DEPTH: continue
        cap = B**(DEPTH - level - 1)
        groups = [np.array([i]) for i in ids] if cap == 1 else (balanced_split(X, ids, cap, rng) if len(ids) > 1 else [ids])
        for c, g in enumerate(groups):
            path[g, level] = c; stack.append((g, level + 1, prefix + [c]))
    codes = path @ (B**np.arange(DEPTH - 1, -1, -1)); assert len(np.unique(codes)) == V, 'paths must be unique'
    np.savez(out.with_suffix('.npz'), path=path, freq=freq)
    rec = dict(status='completed', tag=a.tag, train_tokens_used=a.tokens, branching=B, depth=DEPTH, vocab=V, unique_paths=int(len(np.unique(codes))), unseen_tokens=int(unseen.sum()),
               wall_s=time.time() - t0, artifact=str(out.with_suffix('.npz').relative_to(ROOT)), artifact_sha256=hashlib.sha256(out.with_suffix('.npz').read_bytes()).hexdigest(),
               source_sha256={n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['experiments/build_vocab_race_tree.py', 'experiments/r1_token_keyed_lm.py']},
               scope='TRAIN-only distributional statistics (first train shard tokens); no validation or test tokens.')
    out.with_suffix('.json').write_text(json.dumps(rec, indent=1) + '\n'); print(json.dumps({k: rec[k] for k in ('unique_paths', 'unseen_tokens', 'wall_s')}))


if __name__ == '__main__':
    main()
