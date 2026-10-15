"""R1 execution: compiled vs eager layer steps on the strongest native learner (5.978388).

Identical restored learner/optimizer/state/RNG/teacher and the identical first ten
updates of the next fresh-TRAIN stream, through the published raw_step. Records
per-update losses and wall time; compile/recompile updates are reported separately.
"""
import argparse, copy, json, platform, resource, sys, time, hashlib
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'experiments'), str(ROOT / 'experiments/credit')]
import episodic_memory_continuation as run
from sleeping_machines.multidepth_ordered_episodes_compiled import token_features as compiled_capable


def main():
    p = argparse.ArgumentParser(); p.add_argument('--tag', required=True); p.add_argument('--updates', type=int, default=10); a = p.parse_args()
    torch.set_num_threads(1); out = {}
    for mode in ('eager', 'compiled'):
        model, args, train = run.build(); args.compiled = mode == 'compiled'
        if args.compiled: run.driver.token_features = compiled_capable
        ck = torch.load(run.PARENT.with_suffix('.pt'), weights_only=False); model.load_state_dict(ck['model'])
        opt = run.gate.CalibratedReceiverBankAdamW(model, ck['optimizer']['bank_caps'], lr=args.lr, weight_decay=args.weight_decay); opt.load_state_dict(copy.deepcopy(ck['optimizer']))
        for n, r in ck['owned_rng'].items(): getattr(model, n).set_state(r)
        gen = torch.Generator().set_state(ck['generator']); teacher = torch.Generator().set_state(ck['teacher_generator']); torch.set_rng_state(ck['torch_rng'])
        state = run.gate.pilot.clone_state(ck['state'])
        start = ck['binding']['stream_start'] + run.TOTAL * args.chunk * args.lanes + args.lanes
        tok = run.driver.LaneTokens(train, start, a.updates * args.chunk * args.lanes + args.lanes, args.lanes)
        losses, times = [], []
        for i in range(a.updates):
            t = time.perf_counter(); opt.zero_grad()
            loss, state, _ = run.gate.pilot.raw_step(model, state, gen, teacher, args, tok[:, i * args.chunk:(i + 1) * args.chunk + 1], 0)
            opt.step(); times.append(time.perf_counter() - t); losses.append(float(loss.double().mean()))
        out[mode] = dict(losses=losses, seconds=times, weights_sha256=hashlib.sha256(b''.join(x.detach().numpy().tobytes() for x in model.parameters())).hexdigest())
    e, c = out['eager'], out['compiled']
    steady = sorted(c['seconds'])[:len(c['seconds']) - 2]; eager_steady = sorted(e['seconds'])[:len(e['seconds']) - 2]
    record = dict(status='completed', battle='R1', tag=a.tag, runs=out,
                  max_loss_difference=max(abs(x - y) for x, y in zip(e['losses'], c['losses'])),
                  eager_steady_s_per_update=sum(eager_steady) / len(eager_steady), compiled_steady_s_per_update=sum(steady) / len(steady),
                  compile_overhead_s=sum(c['seconds']) - sum(steady) - (len(c['seconds']) - len(steady)) * sum(steady) / len(steady),
                  host=platform.node(), max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  source_sha256={n: run.driver.sha(ROOT / n) for n in ('analysis/ordered_compiled_equivalence.py', 'sleeping_machines/multidepth_ordered_episodes_compiled.py', 'sleeping_machines/sparse_counterfactual_layer.py')},
                  scope='Execution diagnostic on Curie beside a running one-thread training job (both one thread). Steady state excludes the two slowest (compile/recompile) updates. Numerical agreement is float-level, not bitwise.')
    record['steady_speedup'] = record['eager_steady_s_per_update'] / record['compiled_steady_s_per_update']
    with (ROOT / f'experiments/results/diagnostics/{a.tag}.json').open('x') as f: json.dump(record, f, indent=2); f.write('\n')
    print(json.dumps({k: record[k] for k in ('max_loss_difference', 'eager_steady_s_per_update', 'compiled_steady_s_per_update', 'steady_speedup')}))


if __name__ == '__main__':
    main()
