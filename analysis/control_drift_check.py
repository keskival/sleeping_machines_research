"""R1 diagnostic: does the plain-continuation DEV drift (5.978 -> 6.010 after 64 updates) depend on compiled execution?

Repeats the control arm's first 64 updates (identical restored learner, stream, RNG, published raw_step) eagerly and
compiled, then scores the recorded DEV protocol. No result files of training runs are written or reused.
"""
import argparse, copy, json, sys, time, hashlib
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'experiments'), str(ROOT / 'experiments/credit')]
import episodic_memory_continuation as run
from public_token_scoring import score


def arm(compiled, updates):
    model, args, train = run.build(compiled); ck = torch.load(run.PARENT.with_suffix('.pt'), weights_only=False); model.load_state_dict(ck['model'])
    opt = run.gate.CalibratedReceiverBankAdamW(model, ck['optimizer']['bank_caps'], lr=args.lr, weight_decay=args.weight_decay); opt.load_state_dict(copy.deepcopy(ck['optimizer']))
    for n, r in ck['owned_rng'].items(): getattr(model, n).set_state(r)
    gen = torch.Generator().set_state(ck['generator']); teacher = torch.Generator().set_state(ck['teacher_generator']); torch.set_rng_state(ck['torch_rng'])
    state = run.gate.pilot.clone_state(ck['state']); start = ck['binding']['stream_start'] + run.TOTAL * args.chunk * args.lanes + args.lanes
    tok = run.driver.LaneTokens(train, start, run.TOTAL * args.chunk * args.lanes + args.lanes, args.lanes); losses = []
    for i in range(updates):
        opt.zero_grad(); loss, state, _ = run.gate.pilot.raw_step(model, state, gen, teacher, args, tok[:, i * args.chunk:(i + 1) * args.chunk + 1], 0); opt.step(); losses.append(float(loss.double().mean()))
    return dict(dev=score(model, run.driver.load_tokens(args.dev_file), args, run.driver, 25165824, 8192, 1024, 8)['nll'], prequential=sum(losses) / len(losses))


def main():
    p = argparse.ArgumentParser(); p.add_argument('--tag', required=True); p.add_argument('--updates', type=int, default=64); a = p.parse_args(); torch.set_num_threads(1); began = time.monotonic()
    out = dict(eager=arm(False, a.updates), compiled=arm(True, a.updates))
    rec = dict(status='completed', battle='R1', tag=a.tag, updates=a.updates, parent_dev=5.978387557528713, recorded_compiled_smoke_dev=json.loads((ROOT / 'experiments/results/token_language/curie_episodic_memory_smoke_none_20261015_v1.json').read_text())['scored']['nll'], **out,
               source_sha256={n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in ('analysis/control_drift_check.py', 'experiments/episodic_memory_continuation.py')}, process_wall_s=time.monotonic() - began)
    with (ROOT / f'experiments/results/diagnostics/{a.tag}.json').open('x') as f: json.dump(rec, f, indent=2); f.write('\n')
    print(json.dumps({k: rec[k] for k in ('parent_dev', 'recorded_compiled_smoke_dev', 'eager', 'compiled')}))


if __name__ == '__main__':
    main()
