#!/usr/bin/env python3
"""B1 task study: structure in the EasyTPP datasets that a mark/time model can exploit (counting only, no training).

Mark side: conditional entropies of the next mark given simple history features, fitted on TRAIN, scored on TEST with
the EasyTPP scored-event convention (events 2..N). Time side: dispersion of inter-event gaps and their dependence on
the previous gap and the current mark. Compare against published per-event mark LL (S2P2 Table 8c).
"""
import json, math
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PUBLISHED_MARK_LL = {'amazon': -1.871, 'retweet': -0.764, 'taxi': -0.211, 'taobao': -1.391, 'stackoverflow': -1.510}
PUBLISHED_TIME_LL = {'amazon': 2.652, 'retweet': -5.584, 'taxi': 0.735, 'taobao': 2.719, 'stackoverflow': -0.641}


def load(ds, split):
    return [(np.array(x['time_since_start'], float), np.array(x['type_event'], int))
            for x in json.load(open(ROOT / f'data/easytpp/{ds}/{split}.json'))]


def cond_ll(train, test, feat, K, alpha=0.5):
    counts, marg = defaultdict(Counter), Counter()
    for t, m in train:
        for i in range(1, len(m)):
            counts[feat(t, m, i)][m[i]] += 1; marg[m[i]] += 1
    n = sum(marg.values()); prior = {k: (marg[k] + 1) / (n + K) for k in range(K)}
    tot, cnt = 0.0, 0
    for t, m in test:
        for i in range(1, len(m)):
            c = counts.get(feat(t, m, i)); s = sum(c.values()) if c else 0
            p = ((c[m[i]] if c else 0) + alpha * K * prior[m[i]]) / (s + alpha * K)
            tot += math.log(p); cnt += 1
    return tot / cnt


def main():
    for ds in PUBLISHED_MARK_LL:
        tr, te = load(ds, 'train'), load(ds, 'test')
        K = max(int(m.max()) for _, m in tr + te) + 1
        rep1 = np.mean([m[i] == m[i - 1] for _, m in te for i in range(1, len(m))])
        rep5 = np.mean([m[i] in set(m[max(0, i - 5):i]) for _, m in te for i in range(1, len(m))])
        rows = {
            'unigram': cond_ll(tr, te, lambda t, m, i: 0, K),
            'prev mark': cond_ll(tr, te, lambda t, m, i: m[i - 1], K),
            'prev 2 marks': cond_ll(tr, te, lambda t, m, i: tuple(m[max(0, i - 2):i]), K),
            'prev mark + gap bin': cond_ll(tr, te, lambda t, m, i: (m[i - 1], int(np.clip(np.log2(t[i] - t[i - 1] + 1e-9), -12, 12))), K),
        }
        gaps = np.concatenate([np.diff(t) for t, _ in te])
        pos = gaps[gaps > 0]
        lag = np.concatenate([np.diff(t) for t, _ in tr])
        a, b = np.log(lag[:-1] + 1e-6), np.log(lag[1:] + 1e-6)
        print(f'\n{ds}: K={K}  P(next=prev)={rep1:.3f}  P(next in last5)={rep5:.3f}')
        for k, v in rows.items():
            print(f'  mark LL {k:<22} {v:8.4f}   (published best {PUBLISHED_MARK_LL[ds]})')
        print(f'  gaps: zero {np.mean(gaps <= 0):.3%}  log-gap mean {np.log(pos).mean():.3f} sd {np.log(pos).std():.3f}  '
              f'corr(log gap, next log gap)={np.corrcoef(a, b)[0, 1]:.3f}  published best time LL {PUBLISHED_TIME_LL[ds]}')
        q = np.percentile(pos, [1, 10, 50, 90, 99]); print('  gap percentiles 1/10/50/90/99:', np.round(q, 4))


if __name__ == '__main__':
    main()
