#!/usr/bin/env python3
"""B1: DEV-vs-TEST difficulty offsets from fixed count models fitted on TRAIN (no learned model, no selection on TEST).

Mark: prev-2-mark conditional counts. Time: log-normal gap density per previous mark (exact per-event time LL with a
zero-gap point mass excluded). Offsets let DEV development numbers be read against TEST-only published references."""
import json, math
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def load(ds, split):
    return [(np.array(x['time_since_start'], float), np.array(x['type_event'], int))
            for x in json.load(open(ROOT / f'data/easytpp/{ds}/{split}.json'))]


def main():
    rows = {}
    for ds in ['amazon', 'retweet', 'taxi', 'taobao', 'stackoverflow']:
        tr = load(ds, 'train'); K = max(int(m.max()) for _, m in tr) + 1
        cnt, marg = defaultdict(Counter), Counter(); gaps = defaultdict(list)
        for t, m in tr:
            for i in range(1, len(m)):
                cnt[tuple(m[max(0, i - 2):i])][m[i]] += 1; marg[m[i]] += 1
                if t[i] > t[i - 1]: gaps[m[i - 1]].append(math.log(t[i] - t[i - 1]))
        n = sum(marg.values()); prior = np.array([(marg[k] + 1) / (n + K) for k in range(K)])
        par = {k: (np.mean(v), np.std(v) + 1e-3) for k, v in gaps.items()}
        out = {}
        for split in ['dev', 'test']:
            ml = tl = c = 0
            for t, m in load(ds, split):
                for i in range(1, len(m)):
                    cc = cnt.get(tuple(m[max(0, i - 2):i])); s = sum(cc.values()) if cc else 0
                    ml += math.log(((cc[m[i]] if cc else 0) + 0.5 * K * prior[m[i]]) / (s + 0.5 * K))
                    g = t[i] - t[i - 1]
                    if g > 0:
                        mu, sd = par.get(m[i - 1], (0, 1)); lg = math.log(g)
                        tl += -lg - math.log(sd) - 0.5 * math.log(2 * math.pi) - 0.5 * ((lg - mu) / sd) ** 2
                    c += 1
            out[split] = (ml / c, tl / c)
        rows[ds] = dict(dev_mark=out['dev'][0], test_mark=out['test'][0], dev_time=out['dev'][1], test_time=out['test'][1],
                        mark_offset_test_minus_dev=out['test'][0] - out['dev'][0],
                        time_offset_test_minus_dev=out['test'][1] - out['dev'][1])
        print(ds, {k: round(v, 4) for k, v in rows[ds].items()})
    (ROOT / 'experiments/results/tpp/b1_split_calibration_20261006.json').write_text(json.dumps(rows, indent=1) + '\n')


if __name__ == '__main__':
    main()
