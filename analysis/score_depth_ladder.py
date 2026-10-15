"""R1 depth ladder: score completed residual/serial arms on the shared long-history protocol.

Rebuilds each arm from its final checkpoint (settings bound there), scores the recorded
8192 DEV targets in disjoint 1024-token windows, and records the decision rule:
depth is useful under residual composition if D16res < D8res < D2res and D8res < D8serial.
"""
import argparse, json, sys, time, hashlib
from pathlib import Path
from types import SimpleNamespace
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'experiments')]
import torch
import residual_depth_token_language as driver
from public_token_scoring import score

ARMS = ('d2_residual', 'd8_serial', 'd8_residual', 'd16_residual')


def main():
    p = argparse.ArgumentParser(); p.add_argument('--tag', required=True); p.add_argument('--prefix', default='curie_depth_ladder'); p.add_argument('--suffix', default='1m_20261015_v1')
    a0 = p.parse_args(); torch.set_num_threads(1); began = time.monotonic(); rows = {}
    for arm in ARMS:
        tag = f'{a0.prefix}_{arm}_{a0.suffix}'; rec_path = ROOT / f'experiments/results/token_language/{tag}.json'
        if not rec_path.exists(): rows[arm] = dict(tag=tag, status='missing'); continue
        rec = json.loads(rec_path.read_text()); ck = torch.load(ROOT / f'experiments/results/token_language/{tag}.pt', weights_only=False)
        args = SimpleNamespace(**ck['settings'])
        tr = driver.load_tokens(args.train_file); train = driver.BlockLaneTokens(tr, args.train_tokens, args.lanes, args.exposure, args.steps, args.chunk)
        counts = train.counts(); order = torch.argsort(counts, descending=True, stable=True)
        torch.manual_seed(args.seed); model = driver.Model(args, order, counts); model.load_state_dict(ck['model'])
        for name in ('site_generator', 'position_generator', 'local_generator', 'alternative_generator'): getattr(model, name).set_state(ck[name])
        scored = score(model, driver.load_tokens(args.dev_file), args, driver, 25165824, 8192, 1024, 8)
        last = rec['curve'][-1]
        rows[arm] = dict(tag=tag, status='completed', depth=args.depth, residual=bool(getattr(args, 'residual', False)), long_history_nll=scored['nll'],
                         final_short_dev_nll=last['dev_nll'], route_entropy=last.get('route_entropy'), memory_effective_rank=last.get('memory_effective_rank'),
                         train_wall_s=rec['train_wall_s_total'], parameters=rec['parameters'],
                         residual_gain_mean=[float(x) for x in model.core.residual_gain.mean(1)] if hasattr(model.core, 'residual_gain') else None)
        print(json.dumps({k: v for k, v in rows[arm].items() if k not in ('memory_effective_rank',)}), flush=True)
    done = all(r['status'] == 'completed' for r in rows.values())
    nll = {k: r.get('long_history_nll') for k, r in rows.items()}
    decision = dict(residual_monotone=done and nll['d16_residual'] < nll['d8_residual'] < nll['d2_residual'],
                    residual_beats_serial_d8=done and nll['d8_residual'] < nll['d8_serial'],
                    d8_residual_gain_over_serial=(nll['d8_serial'] - nll['d8_residual']) if done else None)
    out = dict(status='completed' if done else 'partial', battle='R1', tag=a0.tag, rows=rows, decision=decision,
               protocol=dict(targets=8192, target_start=25165825, window=1024, lanes=8, eos_state_reset=False),
               source_sha256={n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in ('analysis/score_depth_ladder.py', 'experiments/residual_depth_token_language.py', 'sleeping_machines/residual_race_episodes.py')},
               process_wall_s=time.monotonic() - began, scope='Single seed, matched 1M fresh targets and flagship settings per arm; development protocol, no public validation.')
    with (ROOT / f'experiments/results/diagnostics/{a0.tag}.json').open('x') as f: json.dump(out, f, indent=2); f.write('\n')
    print(json.dumps(decision))


if __name__ == '__main__':
    main()
