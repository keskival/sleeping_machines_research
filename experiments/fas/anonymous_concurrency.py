"""Anonymous concurrency estimate for sizing the binding memory (B3; THEORY §434.3). From anonymous training logs only:
the start type is the earliest by mean occurrence time; the end type is the latest among types every item visits (count
within 15% of the start type). Concurrency(t) = starts seen - ends seen. The identity count is printed beside it as a
check (evaluation only)."""
import sys, numpy as np
sys.path.insert(0, 'experiments/fas'); import generate_v2 as V2
runs = [V2.sample(20261005, 'train_clean', r, 2, .02, 0., False) for r in range(60)]
mean_t = {}
for ids, t, *_ in runs:
    for e in np.unique(ids[ids != 0]): mean_t.setdefault(int(e), []).append(t[ids == e].mean())
order = sorted(mean_t, key=lambda e: np.mean(mean_t[e])); start = order[0]
cnt = {e: np.mean([(ids == e).sum() for ids, *_ in runs]) for e in order}
end = [e for e in order if abs(cnt[e] / cnt[start] - 1) < .15][-1]      # last type that every item visits
anon, ident = [], []
for ids, t, line, item, *_ in runs:
    p = ids != 0; e = ids[p]
    c = np.cumsum((e == start).astype(int) - (e == end).astype(int)); anon.append(c.max())
    who = line[p].astype(int) * 1000 + item[p]
    first, last = {}, {}
    for i, w in enumerate(who):
        first.setdefault(w, i); last[w] = i
    ev = np.zeros(len(who))
    for w in first: ev[first[w]] += 1; ev[last[w]] -= 1
    ident.append(np.cumsum(ev).max())
print(f'start type {start}, end type {end} (anonymous, by mean occurrence time)', flush=True)
print(f'max concurrency per log: anonymous mean {np.mean(anon):.1f} max {np.max(anon)} | identity mean {np.mean(ident):.1f} max {np.max(ident)}', flush=True)
