"""R1 history reliance of a completed continuation arm (episodic or plain).

Frozen weights; recorded 8192 DEV targets in disjoint 1024-token windows. Resets the
complete numeric state (and the episodic store, if any) every N observed tokens,
N in {1024 (none), 256, 64, 16}. The N=1024 row must reproduce the arm's recorded
score. Signed loss differences measure reliance on history older than N tokens.
"""
import argparse, json, resource, sys, time
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'experiments'), str(ROOT / 'experiments/credit')]
import episodic_memory_continuation as run
from public_token_scoring import windows, reference_forward
from sleeping_machines.episodic_pointer_memory import EpisodicPointerMemory

SPANS = (1024, 256, 64, 16)


@torch.no_grad()
def score(model, memory, dev, args, span):
    model.eval(); forward = reference_forward(model, run.driver); gen = torch.Generator().manual_seed(args.seed + 100000)
    total = 0.; count = 0; d = model.core.total_payload * model.core.depth
    for inputs, labels, _ in windows(dev, 25165824, 8192, 1024, 8):
        state = None; mem = memory.empty(inputs.shape[0], d) if memory is not None else None
        for pos in range(0, 1024, 16):
            if pos % span == 0 and pos:
                state = None; mem = memory.empty(inputs.shape[0], d) if memory is not None else None
            x, state, _ = forward(inputs[:, pos:pos + 16], state, gen, args, deterministic=args.deterministic_eval)
            losses = model.readout.nll(x, labels[:, pos:pos + 16])
            if memory is not None:
                losses, mem, _ = memory.mixed_nll(losses, x, inputs[:, pos:pos + 16], labels[:, pos:pos + 16], mem)
            total += float(losses.double().sum()); count += losses.numel()
    return total / count


def main():
    p = argparse.ArgumentParser(); p.add_argument('--tag', required=True); p.add_argument('--run', required=True); a = p.parse_args()
    torch.set_num_threads(1); began = time.monotonic()
    record = json.loads((ROOT / f'experiments/results/token_language/{a.run}.json').read_text()); assert record['status'] == 'completed'
    model, args, train = run.build(record['binding'].get('compiled', False))
    ck = torch.load(ROOT / record['final_weights'], weights_only=False); model.load_state_dict(ck['model'])
    for n, r in ck['owned_rng'].items(): getattr(model, n).set_state(r)
    memory = None
    if record['memory'] == 'episodic':
        b = record['binding']; memory = EpisodicPointerMemory(model.core.total_payload * model.core.depth, b['key_dim'], b['capacity'], b['topk'])
        memory.load_state_dict(ck['memory'])
    dev = run.driver.load_tokens(args.dev_file)
    nll = {span: score(model, memory, dev, args, span) for span in SPANS}
    assert abs(nll[1024] - record['scored']['nll']) < 1e-6, (nll[1024], record['scored']['nll'])
    out = dict(status='completed', battle='R1', tag=a.tag, run=a.run, memory=record['memory'], nll={str(k): v for k, v in nll.items()},
               history_value={str(k): nll[k] - nll[1024] for k in SPANS[1:]},
               source_sha256={n: run.driver.sha(ROOT / n) for n in ('analysis/history_reliance.py', 'experiments/episodic_memory_continuation.py', 'sleeping_machines/episodic_pointer_memory.py')},
               max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, process_wall_s=time.monotonic() - began,
               scope='Frozen completed arm; state and store reset every N tokens within the recorded DEV windows. Distribution intervention, not a trained short-history control.')
    with (ROOT / f'experiments/results/diagnostics/{a.tag}.json').open('x') as f: json.dump(out, f, indent=2); f.write('\n')
    print(json.dumps(out['history_value']))


if __name__ == '__main__':
    main()
