"""k-winner write wrapper for the pinned native language driver (FINDINGS 5 Oct, write-bandwidth hypothesis; THEORY §428).

Usage: language_kwrite_benchmark.py --write-k K [--deliver-k] <language_batched_benchmark.py arguments>  (requires --compiled)

The k earliest race arrivals per head all write their slot on every event (sleeping_machines.recruit_layer, write_k).
By default the first arrival alone delivers, so inference stays one delivered value per head. --deliver-k delivers the
mean of the k proposals instead.

The wrapper replaces compiled_episodes.compiled_step for this process, runs language_batched_benchmark.main()
unchanged, and records write_k, deliver_k and its own sha256 in the result JSON. The base driver's bytes are unchanged.

Scope of the work numbers: the traced first windows use the eager batched path (k = 1 operator sequence). The extra k−1
memory writes per head and event are not in the traced estimate; they are small next to proposal computation, but they
are stated, not measured.
"""
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'experiments'))


def main():
    argv = sys.argv[1:]
    if '--write-k' not in argv or '--compiled' not in argv:
        raise ValueError('--write-k and --compiled required')
    i = argv.index('--write-k'); k = int(argv[i + 1]); del argv[i:i + 2]
    deliver = '--deliver-k' in argv
    if deliver:
        argv.remove('--deliver-k')
    tag = argv[argv.index('--tag') + 1]
    import sleeping_machines.compiled_episodes as ce
    from sleeping_machines.recruit_layer import recruit_layer

    def kstep(*args):
        return recruit_layer(*args, write_k=k, deliver_k=deliver)[:7]
    cache = {}

    def compiled_step():
        if 'step' not in cache:
            from torch._dynamo import config as dynamo_config
            for name in ('cache_size_limit', 'recompile_limit'):
                if hasattr(dynamo_config, name):
                    setattr(dynamo_config, name, max(getattr(dynamo_config, name), 64))
            cache['step'] = torch.compile(kstep, dynamic=False, fullgraph=True)
        return cache['step']
    ce.compiled_step = compiled_step
    sys.argv = [str(ROOT / 'experiments/language_batched_benchmark.py'), *argv]
    import language_batched_benchmark
    language_batched_benchmark.main()
    out = ROOT / 'experiments/results/language_batched' / f'{tag}.json'
    if out.exists():
        r = json.loads(out.read_text())
        r['write_k'] = k; r['deliver_k'] = deliver
        r['layer'] = 'sleeping_machines.recruit_layer (k earliest arrivals write) via experiments/language_kwrite_benchmark.py'
        r.setdefault('source_sha256', {})['experiments/language_kwrite_benchmark.py'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        r['source_sha256']['sleeping_machines/recruit_layer.py'] = hashlib.sha256((ROOT / 'sleeping_machines/recruit_layer.py').read_bytes()).hexdigest()
        out.write_text(json.dumps(r, indent=2) + '\n')


if __name__ == '__main__':
    main()
