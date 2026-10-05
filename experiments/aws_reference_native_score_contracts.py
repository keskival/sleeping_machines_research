"""Numerical scorer contracts on synthetic GPT-2 IDs; run through run_safe."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'experiments'))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True); p.add_argument('--tag', required=True)
    a = p.parse_args()
    packet = json.loads(Path(a.manifest).read_text())
    pins = packet['contract_source_sha256']
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    for name, digest in pins.items():
        if sha(ROOT/name) != digest: raise ValueError('Source mismatch: '+name)
    out = ROOT/'experiments/results/diagnostics'/(a.tag+'.json')
    if out.exists(): raise FileExistsError(out)
    import torch
    import numpy as np
    import horizon_token_language_engine as lab
    from aws_reference_target_plan import TargetWindow
    from aws_reference_stream_score import native_window, reduce_windows
    torch.set_num_threads(1); torch.manual_seed(6)
    args = SimpleNamespace(seed=6, payload=4, depth=2, heads=2, pool=3,
        input_init='balanced', tie_pools=False, readout='adaptive',
        readout_init='frequency', minimum_tail_width=0, state_mode='carry',
        route_credit='none', future_site='uniform', free_bias=0., temperature=1.,
        compiled=False, deterministic_eval=False, credit_window=16)
    counts = torch.ones(50257, dtype=torch.long)
    model = lab.Model(args, torch.arange(50257), counts); model.eval()
    data = np.arange(37, dtype=np.int64)+100
    data[[5, 24]] = 50256  # observed document boundary in different sequence groups
    sequence_targets, lanes, batches = 9, 2, 2

    def windows(chunk):
        for batch in range(batches):
            for offset in range(0, sequence_targets, chunk):
                starts = tuple(batch*lanes*sequence_targets+lane*sequence_targets+offset for lane in range(lanes))
                yield TargetWindow(batch, offset, min(chunk, sequence_targets-offset), starts, offset==0)

    def run(chunk):
        gen = torch.Generator().manual_seed(100006)
        finals = []; latest = None
        def callback(window, state):
            nonlocal latest
            loss, count, state = native_window(model, data, args, gen, window, state)
            latest = state
            if window.offset+window.count == sequence_targets: finals.append(state)
            return loss, count, state
        result = reduce_windows(windows(chunk), callback, expected_targets=36,
                                sequence_targets=sequence_targets, batches=batches)
        return result, gen.get_state(), finals

    full, full_rng, full_states = run(9)
    short, short_rng, short_states = run(2)
    assert abs(full['nll']-short['nll']) < 2e-6
    assert torch.equal(full_rng, short_rng)
    for x,y in zip(full_states, short_states):
        for key in ('mem','arr','seen','ctx_vals','ctx_arr','has_ctx','position'):
            xs=x[key] if isinstance(x[key],list) else [x[key]]
            ys=y[key] if isinstance(y[key],list) else [y[key]]
            assert all(torch.equal(xv,yv) for xv,yv in zip(xs,ys)),key
    for name,digest in pins.items():
        if sha(ROOT/name)!=digest:raise ValueError('Source changed: '+name)
    record = dict(status='completed', targets=36, full=full, partitioned=short,
        nll_difference=short['nll']-full['nll'], final_state_exact=True,
        route_rng_exact=True, source_sha256=pins,
        max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope='Initialized integrated P4/D2/H2/U3 model, synthetic GPT-2 IDs with observed EOS, two2-lane9-target sequence groups. Same native_window as public scorer; partition/state/route-RNG contracts only. No fit or public score.')
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)


if __name__=='__main__':main()
