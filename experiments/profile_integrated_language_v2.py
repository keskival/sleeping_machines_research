"""R1 whole-step timing of the retained integrated language recipe.

Execute only through run_safe. Wrappers retain operations, RNG and updates;
timings include wrapper overhead and do not constitute a FLOP accounting.
"""
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import horizon_token_language_engine as engine
from sleeping_machines.token_readout import TokenReadout


def main():
    if '--batched-lookup' in sys.argv:
        sys.argv.remove('--batched-lookup')
        from batched_token_episodes import token_features
        engine.token_features = token_features
    timings = defaultdict(float)
    calls = defaultdict(int)
    originals = []

    def wrap(cls, name, label):
        original = getattr(cls, name)
        originals.append((cls, name, original))

        def timed(self, *args, **kwargs):
            key = label(self)
            start = time.perf_counter()
            try:
                return original(self, *args, **kwargs)
            finally:
                timings[key] += time.perf_counter() - start
                calls[key] += 1
        setattr(cls, name, timed)

    wrap(engine.Model, 'forward', lambda m: 'evaluation_core' if not m.training else
         ('alternative_core' if m.shadow_mode else 'factual_core'))
    wrap(TokenReadout, 'nll', lambda m: 'training_readout' if m.training else 'evaluation_readout')
    wrap(torch.Tensor, 'backward', lambda _: 'backward')
    wrap(torch.optim.AdamW, 'step', lambda _: 'optimizer')
    began = time.perf_counter()
    try:
        engine.main()
    finally:
        for cls, name, original in reversed(originals):
            setattr(cls, name, original)
    tag = sys.argv[sys.argv.index('--tag') + 1]
    path = engine.ROOT / 'experiments/results/token_language' / f'{tag}.json'
    result = json.loads(path.read_text())
    result['profile'] = dict(stage_wall_s=dict(timings), stage_calls=dict(calls),
                            process_wall_s=time.perf_counter()-began,
                            scope='Nonoverlapping instrumented core/readout/backward/optimizer calls; training_readout includes factual and sampled alternative scoring. Other work is retained in full engine wall. Acquisition/data hashes, evaluation and serialization are included only in process wall; no inference of whole-fit FLOPs.')
    result['identity']['source_sha256']['experiments/profile_integrated_language_v2.py'] = engine.sha(__file__)
    result['source_sha256'] = dict(result['identity']['source_sha256'])
    if engine.token_features.__module__ == 'batched_token_episodes':
        result['source_sha256']['experiments/batched_token_episodes.py'] = engine.sha(Path(__file__).with_name('batched_token_episodes.py'))
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'profile': result['profile']}), flush=True)


if __name__ == '__main__':
    main()
