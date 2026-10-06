"""I(next type ; log own duration | current type) on true item tracks (K=1 lines), by event dropout p.
Plug-in estimate with 24 log-duration bins; also H(log-duration bin | current type) for scale; and the shuffle bias
(next types permuted within each current type) as the estimator's null floor."""
import sys; sys.path.insert(0, 'experiments/fas')
import numpy as np, generate_v2 as V2
from collections import defaultdict
def mi(triples, rng=None):
    a, b, d = (np.array(x) for x in zip(*triples))
    if rng is not None:
        b = b.copy()
        for v in np.unique(a): m = a == v; b[m] = rng.permutation(b[m])
    edges = np.quantile(d, np.linspace(0, 1, 25)[1:-1]); db = np.searchsorted(edges, d)
    tot = 0.; hd = 0.
    for v in np.unique(a):
        m = a == v; w = m.mean(); bb, dd = b[m], db[m]
        joint = defaultdict(int)
        for x, y in zip(bb, dd): joint[(x, y)] += 1
        n = m.sum(); pb = defaultdict(float); pd = defaultdict(float)
        for (x, y), c in joint.items(): pb[x] += c / n; pd[y] += c / n
        tot += w * sum(c / n * np.log((c / n) / (pb[x] * pd[y])) for (x, y), c in joint.items())
        hd += w * -sum(p * np.log(p) for p in pd.values())
    return tot, hd
rng = np.random.default_rng(0)
for p in (0., .02, .05):
    trip = []
    for r in range(150):
        ids, t, line, item, *_ = V2.sample(20261005, 'train_clean', r, 1, p, 0., False)
        last = {}; prev = {}
        for k in np.flatnonzero(ids != 0):
            i = int(item[k])
            if i in last:
                j = last[i]; trip.append((prev.get(i, -1) * 100 + int(ids[j]), int(ids[k]), np.log(t[k] - t[j] + 1.)))
                prev[i] = int(ids[j])
            last[i] = k
    I, H = mi(trip); I0, _ = mi(trip, rng)
    print(f'drop {p}: I(next type; log dur | prev type, type) = {I:.3f} nats (shuffle floor {I0:.3f}); H(dur bin | prev, type) = {H:.3f}; pairs {len(trip)}')
