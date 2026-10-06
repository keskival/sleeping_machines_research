#!/usr/bin/env python3
"""Count-based n-gram references on the exact tokenized FineWeb slices used by the native token stages.

No neural training. Interpolated absolute-discount (Kneser-Ney style) unigram/bigram/trigram models are fitted on
the first N tokens of the native TRAIN shard and scored on the native DEV interval (8 lanes x 256 tokens from
offset 20,971,520 of the val shard; 255 next-token targets per lane = 2,040 targets), the protocol of
curie_data_growth_tokens_*_p24_s6_20261005_v1 and curie_original_1m_p24_s7_20261006_v1. A larger DEV interval
(8 lanes x 8,192 tokens, same offset) is reported separately to show the evaluation noise of 2,040 targets.
"""
import argparse
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
V = 50257


def load(path):
    header = np.fromfile(path, dtype=np.int32, count=256)
    assert header[0] == 20240520, 'not an llm.c token shard'
    return np.fromfile(path, dtype=np.uint16, offset=1024)


def targets(dev, offset, lanes, lane_tokens):
    block = dev[offset:offset + lanes * lane_tokens].astype(np.int64).reshape(lanes, lane_tokens)
    return [(int(r[i - 2]) if i >= 2 else None, int(r[i - 1]), int(r[i])) for r in block for i in range(1, lane_tokens)]


class KneserNey:
    """Interpolated absolute discounting with continuation-count lower orders."""

    def __init__(self, train, discount):
        self.d = discount
        self.bi = defaultdict(Counter)
        self.tri = defaultdict(Counter)
        xs = train.tolist()
        for a, b in zip(xs[:-1], xs[1:]):
            self.bi[a][b] += 1
        for a, b, c in zip(xs[:-2], xs[1:-1], xs[2:]):
            self.tri[(a, b)][c] += 1
        cont = np.zeros(V)
        for c in self.bi.values():
            for y in c:
                cont[y] += 1
        self.cont = (cont + 0.01) / (cont.sum() + 0.01 * V)
        self.bi_n = {a: sum(c.values()) for a, c in self.bi.items()}
        self.tri_n = {k: sum(c.values()) for k, c in self.tri.items()}

    def p_bigram(self, x, y):
        c = self.bi.get(x)
        if not c:
            return self.cont[y]
        n = self.bi_n[x]
        return max(c[y] - self.d, 0) / n + self.d * len(c) / n * self.cont[y]

    def p_trigram(self, w, x, y):
        lower = self.p_bigram(x, y)
        c = self.tri.get((w, x)) if w is not None else None
        if not c:
            return lower
        n = self.tri_n[(w, x)]
        return max(c[y] - self.d, 0) / n + self.d * len(c) / n * lower


def score(nll):
    a = np.array(nll)
    return dict(nll=float(a.mean()), standard_error=float(a.std(ddof=1) / math.sqrt(len(a))), targets=len(a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--train-file', default='data/fineweb_gpt2/fineweb_train_000001.bin')
    ap.add_argument('--dev-file', default='data/fineweb_gpt2/fineweb_val_000000.bin')
    ap.add_argument('--sizes', default='65536,262144,1048576,4194304')
    ap.add_argument('--discount', type=float, default=0.9)
    a = ap.parse_args()
    start = time.time()
    train_all, dev = load(ROOT / a.train_file), load(ROOT / a.dev_file)
    native_dev = targets(dev, 20971520, 8, 256)
    large_dev = targets(dev, 20971520, 8, 8192)
    rows = []
    for n in [int(s) for s in a.sizes.split(',')]:
        train = train_all[:n].astype(np.int64)
        counts = np.bincount(train, minlength=V).astype(float)
        unigram = (counts + 1) / (n + V)
        model = KneserNey(train, a.discount)
        row = dict(train_tokens=n)
        for name, dev_targets in (('native_dev_2040', native_dev), ('large_dev_65528', large_dev)):
            row[name] = dict(
                unigram_add1=score([-math.log(unigram[y]) for _, _, y in dev_targets]),
                bigram_kn=score([-math.log(model.p_bigram(x, y)) for _, x, y in dev_targets]),
                trigram_kn=score([-math.log(model.p_trigram(w, x, y)) for w, x, y in dev_targets]))
        rows.append(row)
        print(json.dumps(row), flush=True)
    source = Path(__file__).resolve()
    result = dict(
        status='completed', tag=a.tag, kind='count_ngram_reference',
        protocol=dict(tokenizer='GPT-2 BPE (llm.c FineWeb shards)', train_file=a.train_file, dev_file=a.dev_file,
                      native_dev=dict(offset=20971520, lanes=8, lane_tokens=256, targets=len(native_dev)),
                      large_dev=dict(offset=20971520, lanes=8, lane_tokens=8192, targets=len(large_dev)),
                      discount=a.discount, units='nats per token'),
        rows=rows, wall_s=time.time() - start,
        source_sha256={str(source.relative_to(ROOT)): hashlib.sha256(source.read_bytes()).hexdigest()})
    out = ROOT / 'experiments/results/token_language' / (a.tag + '.json')
    out.write_text(json.dumps(result, indent=2) + '\n')
    print('wrote', out)


if __name__ == '__main__':
    main()
