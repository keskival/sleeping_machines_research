import sys, math, numpy as np
sys.path.insert(0, 'experiments/fas'); import binding_toy as B
rng = np.random.default_rng(0); K, R, cv, stagger = 3, 8, .15, 8.
medians = np.exp(np.linspace(math.log(3.), math.log(20.), R))[rng.permutation(R)]
data = [B.sample(rng, K, R, medians, cv, stagger, 0.) for _ in range(320)][256:]
same = n = 0; same_true = 0
for e, t, w in data:
    # tracks: true-model greedy assignment; a track = (next step index, last time)
    tracks = []; assign = []
    for ei, ti in zip(e, t):
        r = ei - 1
        best, bl = None, -1e9
        for j, (nxt, lt) in enumerate(tracks):
            if nxt == r:
                d = ti - lt; l = -((math.log(max(d, 1e-6)) - math.log(medians[r])) ** 2) / (2 * cv * cv) - math.log(max(d, 1e-6))
                if l > bl: bl, best = l, j
        if r == 0 or best is None:
            tracks.append([r + 1, ti]); assign.append(len(tracks) - 1)
        else:
            tracks[best] = [r + 1, ti]; assign.append(best)
    a = np.array(assign)
    for s in np.unique(a):
        x = w[a == s]; same += int((x[1:] == x[:-1]).sum()); n += len(x) - 1
print('Bayes-greedy binding alpha (true routes and laws):', round(same / n, 3))
