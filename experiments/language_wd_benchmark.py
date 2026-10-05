"""Weight-decay wrapper for the pinned native language driver (THEORY §421): AdamW instead of Adam.

Usage: language_wd_benchmark.py --weight-decay WD <language_batched_benchmark.py arguments>

The wrapper removes --weight-decay from argv, replaces torch.optim.Adam with AdamW(weight_decay=WD) for the
driver's single optimizer, runs language_batched_benchmark.main() unchanged, and then records `weight_decay` and this
wrapper's sha256 in the result JSON. The base driver's bytes are unchanged (its hash is pinned by AWS manifests).
"""
import functools
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments'))


def main():
    argv = sys.argv[1:]
    if '--weight-decay' not in argv:
        raise ValueError('--weight-decay required')
    i = argv.index('--weight-decay'); wd = float(argv[i + 1]); del argv[i:i + 2]
    tag = argv[argv.index('--tag') + 1]
    sys.argv = [str(ROOT / 'experiments/language_batched_benchmark.py'), *argv]
    torch.optim.Adam = functools.partial(torch.optim.AdamW, weight_decay=wd)
    import language_batched_benchmark
    language_batched_benchmark.main()
    out = ROOT / 'experiments/results/language_batched' / f'{tag}.json'
    r = json.loads(out.read_text())
    r['weight_decay'] = wd; r['optimizer'] = 'AdamW (decoupled weight decay) via experiments/language_wd_benchmark.py'
    r.setdefault('source_sha256', {})['experiments/language_wd_benchmark.py'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    out.write_text(json.dumps(r, indent=2) + '\n')


if __name__ == '__main__':
    main()
