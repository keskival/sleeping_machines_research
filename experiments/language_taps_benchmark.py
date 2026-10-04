"""Native language benchmark with content taps: each character event carries the one-hot of the last --taps characters
(current first), as the Mackey-Glass member carries its last samples (FINDINGS.md, 5 October delay-tap hypothesis).

Motivation: the tuned-LSTM lead at 10M is present at 2-4 characters of history (curie 23003f5d); in the native model
recent identity reaches the current event only through race-selected writes and the previous winner value.  Taps keep
races, sparse addressed writes and route credit unchanged; they widen only the event content projection (27*k inputs).
Causal: taps read text before the event (zeros before position 0 of the text).

Runs experiments/language_batched_benchmark.py unchanged (its pinned bytes are not edited): this wrapper replaces its
rows_of and the content width, then calls its main().  Use --taps 1 to reproduce the base driver's events exactly.
"""
import sys

import numpy as np

import language_batched_benchmark as base

K = 1
if '--taps' in sys.argv:
    i = sys.argv.index('--taps'); K = int(sys.argv[i + 1]); del sys.argv[i:i + 2]
EYE = np.eye(27, dtype=np.float32)


def rows_of(text, starts, S):
    rows = []
    for s in starts:
        ev = []
        for t in range(S):
            parts = [EYE[text[s + t - j]] if s + t - j >= 0 else np.zeros(27, np.float32) for j in range(K)]
            ev.append((float(t), np.concatenate(parts)))
        rows.append(dict(events=ev))
    return rows


class TappedHeads(base.AddressedEventHeads):
    def __init__(self, *args, content_dim=27, **kw):
        super().__init__(*args, content_dim=27 * K, **kw)


base.rows_of = rows_of
base.AddressedEventHeads = TappedHeads

if __name__ == '__main__':
    base.main()
