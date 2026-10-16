#!/usr/bin/env python3
"""r1_deep_keyed_stream.py unchanged, with the temporal-memory kernel chunk set by --tm-chunk (execution only).

ParallelTemporalMemory computes the exact causal kernel in chunks with the complex state carried across chunk boundaries,
so any chunk gives the same function (TM output agreement 1e-6 relative; loss and gradients in
curie_tm_chunk_equivalence_20261016_v1). Chunk 64 builds a 16x smaller kernel than the default 256 and ran a CPU L2
update 3.3x faster (0.30 s vs 0.99 s at batch 1). The completed result gains an 'execution' record naming the chunk
and this wrapper's hash; every other field is written by the unchanged driver.
"""
import hashlib, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path[:0] = [str(ROOT/'experiments'), str(ROOT/'experiments/tpp')]
from r1_deep_keyed_lm import ParallelTemporalMemory  # noqa: E402
import r1_deep_keyed_stream as stream  # noqa: E402

if __name__ == '__main__':
    argv = sys.argv[1:]; chunk = 256
    if '--tm-chunk' in argv:
        i = argv.index('--tm-chunk'); chunk = int(argv[i + 1]); del argv[i:i + 2]
    ParallelTemporalMemory.CHUNK = chunk; sys.argv = [sys.argv[0]] + argv
    stream.main()
    tag = argv[argv.index('--tag') + 1]; out = ROOT/f'experiments/results/token_language/{tag}.json'
    if out.exists():
        r = json.loads(out.read_text())
        if 'execution' not in r:
            r['execution'] = dict(tm_chunk=chunk, wrapper='experiments/r1_deep_keyed_stream_chunked.py', wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
            out.write_text(json.dumps(r, indent=1) + '\n')
