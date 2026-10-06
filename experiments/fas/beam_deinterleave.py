"""Beam de-interleaver for FAS v2: STRUCTURE-ASSISTED DIAGNOSTIC, not an information-matched reference
(THEORY note 155 §432; user direction 6 October 2026: knowing the processes is privileged knowledge).

Its parameters are fitted on anonymous merged logs only, but its design used privileged process knowledge. The
rank-alignment initialisation, the irregular-step insertion rule and the tie handling were designed after inspecting
true item routes from the identity sidecar (one route per item, a repeated step, an optional retry step, same-ms
ties). Exclude it from fair-reference selection and from the Stage 1 gate. Development stopped on 6 October 2026,
unfinished:
- pair accuracy against identity on 10 validation logs (40 fit logs) was ~0.49 at K=1, drop 0 and ~0.26 at K=2,
  drop .02;
- for comparison, the same inference with parameters from true tracks reached 0.96 and 0.62.

Model. Every process event belongs to one track (one item on one line). A track starts with type b at rate rho*start[b],
moves a -> b with probability T[a, b] (learned, mixed with a uniform backoff of 1%) after a gap whose log(1 + gap_ms) is Gaussian per (a, b), and ends after type e when
end[e] > 1/2. The event's source is uniform over the open tracks. A beam over track assignments (width B; each
hypothesis is a set of open tracks with their last type and time) keeps the B best cumulative log-likelihoods.

Learning (hard EM on anonymous training logs):
- Initial tracks by rank alignment (init_from_ranks): the r-th occurrence of each type is assigned to the r-th item.
- Each iteration tracks the training logs and re-estimates start, T, end, rho and the gap densities from the best
  hypothesis's tracks.

Scoring, as oracle_bound.py does with true identity: item-own durations of the best hypothesis's tracks at prefix N,
robust z against the learned clean durations per (a, b). Outputs signed and squared means, max_step, and the tracker's
negative log-likelihood per event.
"""
from collections import defaultdict

import numpy as np

V = 46
NEG = -1e30


class BeamTracker:
    def __init__(self, slots=48, width=16, eps=1e-4, backoff=.01):
        self.slots, self.width, self.eps, self.backoff = slots, width, eps, backoff

    # ---------------------------------------------------------------- learning
    def init_from_order(self, train):
        """anonymous initial model: types sorted by mean occurrence time (each item visits each route type, so the mean
        occurrence time is a constant plus the type's mean offset in the route); T0 favours the next few types in that
        order; gaps start uninformative. rho is fixed at the anonymous item count (median per-type count) per event."""
        mean_t = defaultdict(list)
        for ids, t in train:
            for e in np.unique(ids[ids != 0]):
                mean_t[int(e)].append(float(t[ids == e].mean()))
        order = sorted(mean_t, key=lambda e: np.mean(mean_t[e]))
        rank = {e: r for r, e in enumerate(order)}
        T = np.full((V, V), self.eps)
        for a in order:
            for b in order:
                d = rank[b] - rank[a]
                if 1 <= d <= 3:
                    T[a, b] = np.exp(-(d - 1))
        self.T = np.log(T / T.sum(1, keepdims=True))
        start = np.full(V, self.eps); start[order[0]] = 1.
        self.start = np.log(start / start.sum())
        self.end = np.zeros(V, bool); self.end[order[-1]] = True
        counts = [np.median(np.unique(ids[ids != 0], return_counts=True)[1]) for ids, _ in train]
        n_ev = sum(int((ids != 0).sum()) for ids, _ in train)
        self.rho_fixed = float(np.sum(counts)) / max(n_ev, 1); self.rho = self.rho_fixed
        lg = np.concatenate([np.log1p(np.diff(t[ids != 0])) for ids, t in train])
        self.mu = np.full((V, V), np.median(lg)); self.sigma = np.full((V, V), max(lg.std(), .1))
        self.dur = {}
        return self

    def init_from_ranks(self, train, excess_min=.1):
        """anonymous initial tracks.
        1. Rank alignment: in each log the r-th occurrence of every unit type (count within 15% of the median per-type
           count, the anonymous item count) is assigned to the r-th item; items rarely overtake.
        2. Irregular types x (repeated or optional steps) are placed by timing. For each consecutive link (a, b) of the
           rank tracks, take the first x event in [t_a, t_b) (same-item events often share a timestamp) and its delay
           from t_a. A step coupled to the item has a consistent delay; a chance hit from another item on a paced
           line does not. Score = hit fraction x fraction of delays within 5% of the link's median span of their
           median. x occupies the ceil(count_x / items) best links (score >= excess_min); those links take the earliest
           unused x event inside their interval.
        The start rate rho stays fixed at items / process events, so hard EM cannot merge items into longer tracks."""
        logs, n_items, n_ev = [], 0, 0
        for ids, t in train:
            proc = ids != 0; ev, tm = ids[proc].astype(np.int64), t[proc].astype(np.float64)
            types, counts = np.unique(ev, return_counts=True)
            n = int(np.median(counts)); n_items += n; n_ev += len(ev)
            items, irregular = defaultdict(list), {}
            for e, c in zip(types, counts):
                where = np.flatnonzero(ev == e)
                if abs(c / n - 1) > .15:
                    irregular[int(e)] = tm[where]
                    continue
                for r, k in enumerate(where):
                    items[min(r, n - 1)].append((int(e), float(tm[k])))
            span = max(tm[-1] - tm[0], 1.) if len(tm) else 1.
            logs.append(([sorted(v, key=lambda x: x[1]) for v in items.values()], irregular, span))
        delays, total = defaultdict(list), defaultdict(int)
        for tracks, irregular, span in logs:
            for x, times in irregular.items():
                for tr in tracks:
                    for (a, ta), (b, tb) in zip(tr[:-1], tr[1:]):
                        j = np.searchsorted(times, ta, 'left'); total[(x, a, b)] += 1
                        if j < len(times) and times[j] < tb:
                            delays[(x, a, b)].append((times[j] - ta, tb - ta))
        per_item = defaultdict(list)
        for tracks, irregular, _ in logs:
            for x, times in irregular.items():
                per_item[x].append(len(times) / max(len(tracks), 1))
        links = defaultdict(set)
        for x in per_item:
            ranked = []
            for key, v in delays.items():
                if key[0] != x:
                    continue
                d, span_ab = np.asarray(v).T
                tol = np.maximum(.05 * np.median(span_ab), 1.)
                ranked.append((len(v) / total[key] * float((np.abs(d - np.median(d)) <= tol).mean()), key[1:]))
            for sc, link in sorted(ranked)[::-1][:int(np.ceil(np.mean(per_item[x]) - .15))]:
                if sc >= excess_min:
                    links[x].add(link)
        self.inserted_links = {x: sorted(v) for x, v in links.items()}
        tracks_all = []
        for tracks, irregular, _ in logs:
            used = {x: np.zeros(len(v), bool) for x, v in irregular.items()}
            out = []
            for tr in tracks:
                new = [tr[0]]
                for (a, ta), (b, tb) in zip(tr[:-1], tr[1:]):
                    for x, times in irregular.items():
                        if (a, b) in links.get(x, ()):
                            lo, hi = np.searchsorted(times, ta, 'left'), np.searchsorted(times, tb, 'left')
                            free = [j for j in range(lo, hi) if not used[x][j]]
                            if free:
                                used[x][free[0]] = True; new.append((x, float(times[free[0]])))
                    new.append((b, tb))
                out.append(sorted(new, key=lambda z: z[1]))
            tracks_all += out
        self.rho_fixed = n_items / max(n_ev, 1)
        self.m_step(tracks_all)
        return self

    def m_step(self, tracks):
        start = np.full(V, self.eps); T = np.full((V, V), self.eps); last = np.zeros(V); seen = np.zeros(V)
        gaps = defaultdict(list); durs = defaultdict(list); n_ev = 0
        for tr in tracks:
            ty = [e for e, _ in tr]; tm = [x for _, x in tr]; n_ev += len(tr)
            start[ty[0]] += 1; last[ty[-1]] += 1
            for e in ty:
                seen[e] += 1
            for (a, ta), (b, tb) in zip(tr[:-1], tr[1:]):
                T[a, b] += 1; gaps[(a, b)].append(np.log1p(tb - ta)); durs[(a, b)].append(tb - ta)
        T = (1 - self.backoff) * T / T.sum(1, keepdims=True) + self.backoff / (V - 1)
        self.start = np.log(start / start.sum()); self.T = np.log(T)
        self.end = last > .5 * np.maximum(seen, 1)
        self.rho = getattr(self, 'rho_fixed', None) or len(tracks) / max(n_ev, 1)
        pooled = np.concatenate([np.asarray(v) for v in gaps.values()]) if gaps else np.zeros(1)
        self.mu = np.full((V, V), np.median(pooled)); self.sigma = np.full((V, V), max(pooled.std(), .1))
        for (a, b), v in gaps.items():
            v = np.asarray(v)
            if len(v) >= 5:
                self.mu[a, b] = np.median(v); self.sigma[a, b] = max(1.4826 * np.median(np.abs(v - np.median(v))), .05)
        self.dur = {k: (float(np.median(v)), max(1.4826 * float(np.median(np.abs(np.asarray(v) - np.median(v)))), 1.0))
                    for k, v in durs.items() if len(v) >= 5}
        return self

    def fit(self, train, iters=3, width=None, init='order'):
        getattr(self, f'init_from_{init}')(train)
        for _ in range(iters):
            tracks = []
            for ids, t in train:
                tracks += self.tracks(ids, t, width=width)
            self.m_step(tracks)
        return self

    # ---------------------------------------------------------------- inference
    def run(self, ids, t, width=None):
        """beam over process events; returns (history, cumulative scores per step, process-event positions)."""
        W = width or self.width; S = self.slots
        pos = np.flatnonzero(ids != 0); ev = ids[pos].astype(np.int64); tm = t[pos].astype(np.float64)
        last_type = np.full((1, S), -1); last_time = np.zeros((1, S)); score = np.zeros(1)
        hist, best, self._step_scores = [], [], []
        log_birth = np.log(self.rho) + self.start
        log_cont = np.log1p(-min(self.rho, .999))
        for e, x in zip(ev, tm):
            open_ = last_type >= 0; n_open = open_.sum(1)
            lt = np.where(open_, last_type, 0)
            g = np.log1p(np.maximum(x - last_time, 0.))
            mu, sg = self.mu[lt, e], self.sigma[lt, e]
            gll = -.5 * ((g - mu) / sg) ** 2 - np.log(sg)
            cont = np.where(open_, self.T[lt, e] + gll + log_cont - np.log(np.maximum(n_open, 1))[:, None], NEG)
            free = ~open_
            # a birth takes the first free slot, or recycles the stalest open track when every slot is in use
            first_free = np.where(free.any(1), free.argmax(1), np.where(open_, last_time, np.inf).argmin(1))
            birth = log_birth[e] - np.log(np.minimum(n_open, S - 1) + 1.)
            cand = score[:, None] + np.concatenate([cont, birth[:, None]], 1)
            flat = cand.ravel()
            k = min(W, int((flat > NEG / 2).sum()))
            top = np.argpartition(-flat, k - 1)[:k] if k < len(flat) else np.arange(len(flat))
            top = top[flat[top] > NEG / 2]
            parent, col = np.divmod(top, S + 1)
            born = col == S
            slot = np.where(born, first_free[parent], col)
            last_type = last_type[parent].copy(); last_time = last_time[parent].copy(); score = flat[top]
            r = np.arange(len(top))
            last_type[r, slot] = -1 if self.end[e] else e
            last_time[r, slot] = x
            hist.append((parent, slot, born)); best.append(score.max()); self._step_scores.append(score)
        self._final = score
        return hist, np.asarray(best), pos

    @staticmethod
    def backtrack(hist, step, j):
        """track label per process event 0..step for hypothesis j at step."""
        slots, borns = [], []
        for parent, slot, born in reversed(hist[:step + 1]):
            slots.append(slot[j]); borns.append(born[j]); j = parent[j]
        slots.reverse(); borns.reverse()
        label, current, counter = np.empty(len(slots), np.int64), {}, 0
        for k, (s, b) in enumerate(zip(slots, borns)):
            if b or s not in current:
                current[s] = counter; counter += 1
            label[k] = current[s]
        return label

    def tracks(self, ids, t, width=None):
        hist, _, pos = self.run(ids, t, width)
        if not hist:
            return []
        label = self.backtrack(hist, len(hist) - 1, int(np.argmax(self._final)))
        out = defaultdict(list)
        for k, l in enumerate(label):
            out[l].append((int(ids[pos[k]]), float(t[pos[k]])))
        return list(out.values())

    def run_scores(self, ids, t, prefixes, width=None):
        hist, best, pos = self.run(ids, t, width)
        out = {}
        for n in prefixes:
            if len(pos) < n:
                out[n] = (np.nan,) * 4; continue
            step = n - 1
            j = int(np.argmax(self._step_scores[step]))
            label = self.backtrack(hist, step, j)
            ev = ids[pos[:n]].astype(np.int64); tm = t[pos[:n]].astype(np.float64)
            lastk, z, per = {}, [], defaultdict(list)
            for k, l in enumerate(label):
                if l in lastk:
                    key = (int(ev[lastk[l]]), int(ev[k]))
                    if key in self.dur:
                        m, s = self.dur[key]; v = float(np.clip((tm[k] - tm[lastk[l]] - m) / s, -50, 50))
                        z.append(v); per[key].append(v)
                lastk[l] = k
            if z:
                z = np.asarray(z)
                out[n] = (float(z.mean()), float((z ** 2).mean()), float(max(np.mean(v) for v in per.values())),
                          float(-best[step] / n))
            else:
                out[n] = (0., 0., 0., float(-best[step] / n))
        return out

