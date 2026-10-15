"""R1: history reliance of a frozen keyed temporal-memory member on the flagship slice.

Targets 25,165,825..25,174,016 of fineweb_val_000000; history restarted every N tokens
(N = 1024 reproduces the cross-member placement). Distribution intervention on frozen weights.
"""
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'experiments')]
import r1_token_keyed_lm as keyed


def main():
    p = argparse.ArgumentParser(); p.add_argument('--tag', required=True); p.add_argument('--run', required=True); a = p.parse_args()
    torch.set_num_threads(1); began = time.monotonic()
    rec = json.loads((ROOT / f'experiments/results/token_language/{a.run}.json').read_text()); g = rec['args']
    model = keyed.KeyedTokenLM(g['d'], g['modes'], g['layers'], g['heads'], g['dk'], g['keyed'], g['dropout'])
    model.load_state_dict(torch.load(ROOT / rec['checkpoint'])); model.eval()
    val = keyed.load(ROOT / 'data/fineweb_gpt2/fineweb_val_000000.bin'); nll = {}
    with torch.no_grad():
        for span in (1024, 256, 64, 16):
            total = n = 0
            for begin in range(25165824, 25165824 + 8192, span):
                t_, n_ = keyed.lane_nll(model, np.asarray(val[begin:begin + span + 1])[None], 256); total += t_; n += n_
            nll[str(span)] = total / n
    out = dict(status='completed', battle='R1', tag=a.tag, run=a.run, nll=nll, history_value={k: nll[k] - nll['1024'] for k in ('256', '64', '16')},
               training_lane=g['lane'], source_sha256={s: hashlib.sha256((ROOT / s).read_bytes()).hexdigest() for s in ('analysis/keyed_history_reliance.py', 'experiments/r1_token_keyed_lm.py')},
               process_wall_s=time.monotonic() - began, scope='Frozen keyed member; history restarted every N tokens on the flagship slice. Intervention, not a trained short-history control.')
    with (ROOT / f'experiments/results/diagnostics/{a.tag}.json').open('x') as f: json.dump(out, f, indent=2); f.write('\n')
    print(json.dumps(out['nll']), json.dumps(out['history_value']))


if __name__ == '__main__':
    main()
