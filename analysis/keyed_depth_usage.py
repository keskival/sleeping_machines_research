"""R1: unique value of each block in the deep keyed member (frozen, no fitting).

For each completed r1_deep_keyed_stream.py run, score the flagship protocol (8192 targets, 1024-token windows) with the
full model and with one residual block removed at a time (the stream passes it unchanged). Loss increase = that block's
unique value. Companion to analysis/depth_usage_probe.py for the race stack (theory/CREDIT_HORIZON_AND_ADDRESSED_MEMORY.md
section 5, P5c).
"""
import argparse, json, sys, time, hashlib
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'experiments'), str(ROOT / 'experiments/tpp')]
from r1_deep_keyed_lm import DeepKeyedLM
from r1_deep_keyed_stream import flagship
from r1_token_keyed_lm import load


def main():
    p = argparse.ArgumentParser(); p.add_argument('--tag', required=True); p.add_argument('--runs', nargs='+', required=True); a0 = p.parse_args()
    torch.set_num_threads(1); began = time.monotonic(); val = load(ROOT / 'data/fineweb_gpt2/fineweb_val_000000.bin'); rows = {}
    for run in a0.runs:
        rec = json.loads((ROOT / f'experiments/results/token_language/{run}.json').read_text()); g = rec['args']
        model = DeepKeyedLM(g['d'], g['modes'], g['layers'], g['heads'], g['dk'], g['dropout']); model.load_state_dict(torch.load(ROOT / rec['checkpoint'])); model.eval()
        full = flagship(model, val)['nll']; assert abs(full - rec['flagship_protocol']['nll']) < 1e-6, (full, rec['flagship_protocol']['nll'])
        blocks = model.blocks; skip = []
        for b in range(len(blocks)):
            model.blocks = torch.nn.ModuleList(blk for i, blk in enumerate(blocks) if i != b); skip.append(flagship(model, val)['nll'] - full)
        model.blocks = blocks
        rows[run] = dict(layers=g['layers'], flagship_nll=full, skip_block_loss_increase=skip)
        print(run, round(full, 4), [round(v, 3) for v in skip], flush=True)
    out = dict(status='completed', battle='R1', tag=a0.tag, rows=rows, protocol=dict(targets=8192, target_start=25165825, window=1024, ablation='block removed from the residual stream'),
               source_sha256={n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in ('analysis/keyed_depth_usage.py', 'experiments/r1_deep_keyed_lm.py', 'experiments/r1_deep_keyed_stream.py')},
               process_wall_s=time.monotonic() - began, scope='Frozen single-seed one-pass deep keyed runs; flagship protocol.')
    with (ROOT / f'experiments/results/diagnostics/{a0.tag}.json').open('x') as f: json.dump(out, f, indent=2); f.write('\n')


if __name__ == '__main__':
    main()
