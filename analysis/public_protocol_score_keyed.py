"""R1: score a plan-B deep keyed member or the multi-epoch keyed member on the pinned public reference's validation protocol (modded-NanoGPT record
95a9fd44, experiments/PUBLIC_LANGUAGE_ADVANTAGE_TARGET.md): fineweb_val_000000 targets 1..10,485,760 in disjoint
1024-token windows, every window-final target counted, state reset only at window starts (no EOS reset; the reference
Transformer attends across EOS inside its window).

The public range [0, 10,485,761) is sealed: development slices start at 10,485,760 (selection), 20,971,520 (65,528) and
25,165,824 (flagship). --mode contract scores development windows only and checks this scorer against the driver's
flagship() on the same windows (must agree to float rounding), then reports s/window for cost planning. --mode public
scores the sealed range once for an explicitly selected model; it refuses to overwrite an existing result.
"""
import argparse, hashlib, json, resource, sys, time
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path[:0] = [str(ROOT/'experiments'), str(ROOT/'experiments/tpp')]
from r1_token_keyed_lm import load, lane_nll, KeyedTokenLM  # noqa: E402
from r1_deep_keyed_lm import DeepKeyedLM  # noqa: E402
import r1_deep_keyed_stream as stream  # noqa: E402

PUBLIC_TARGETS, WINDOW = 10485760, 1024


def score(model, val, start, targets, lanes):
    """Mean NLL over targets start+1 .. start+targets in disjoint WINDOW-token windows, `lanes` windows per batch."""
    assert targets % WINDOW == 0 and start + targets < len(val)
    model.eval(); total = n = 0; began = time.time()
    with torch.no_grad():
        for b in range(start, start + targets, WINDOW*lanes):
            k = min(lanes, (start + targets - b)//WINDOW)
            w = np.stack([np.asarray(val[p:p + WINDOW + 1]) for p in range(b, b + k*WINDOW, WINDOW)])
            t_, n_ = lane_nll(model, w, 256); total += t_; n += n_
    assert n == targets
    return dict(nll=total/n, loss_sum=total, targets=n, target_start=start + 1, target_end_inclusive=start + targets, window=WINDOW,
                eos_state_reset=False, s_per_window=(time.time() - began)/(targets//WINDOW))


def main():
    p = argparse.ArgumentParser(); p.add_argument('--tag', required=True); p.add_argument('--run', required=True)
    p.add_argument('--mode', choices=('contract', 'public'), required=True); p.add_argument('--lanes', type=int, default=8); a = p.parse_args()
    torch.set_num_threads(1); began = time.time()
    rec = json.loads((ROOT/f'experiments/results/token_language/{a.run}.json').read_text()); g = rec['args']; assert rec.get('status', 'completed') == 'completed'
    model = (KeyedTokenLM(g['d'], g['modes'], g['layers'], g['heads'], g['dk'], g['keyed'], g['dropout']) if 'keyed' in g   # multi-epoch keyed member
             else DeepKeyedLM(g['d'], g['modes'], g['layers'], g['heads'], g['dk'], g['dropout'])); model.load_state_dict(torch.load(ROOT/rec['checkpoint'])); model.eval()
    val = load(ROOT/'data/fineweb_gpt2/fineweb_val_000000.bin'); out = dict(battle='R1', tag=a.tag, run=a.run, mode=a.mode, parameters=rec['parameters'], presentations=rec.get('presentations'), best_epoch=rec.get('best_epoch'),
                                                                         train_tokens=g['train_tokens'], layers=g['layers'], d=g['d'])
    if a.mode == 'contract':
        mine = score(model, val, stream.FLAGSHIP_START, stream.FLAGSHIP_TARGETS, a.lanes); ref = stream.flagship(model, val)
        out.update(development=mine, driver_flagship=ref, abs_diff=abs(mine['nll'] - ref['nll']), recorded_flagship=(rec.get('flagship_protocol') or {}).get('nll'),
                   public_cost_estimate_h=mine['s_per_window']*(PUBLIC_TARGETS//WINDOW)/3600)
        out['status'] = 'completed' if out['abs_diff'] < 1e-6 and (out['recorded_flagship'] is None or abs(mine['nll'] - out['recorded_flagship']) < 1e-6) else 'failed'
        path = ROOT/f'experiments/results/diagnostics/{a.tag}.json'
    else:
        out.update(public=score(model, val, 0, PUBLIC_TARGETS, a.lanes), reference=dict(record='modded-nanogpt 2024-10-29 optimizer record 95a9fd44',
                   local='experiments/results/public_reference/modded_nanogpt_20241029_optimizer_record.txt', final_val_nll=3.3406, tokens_per_step=524288))
        out['status'] = 'completed'; path = ROOT/f'experiments/results/token_language/{a.tag}.json'
    out.update(max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, process_wall_s=time.time() - began,
               source_sha256={n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ('analysis/public_protocol_score_keyed.py', 'experiments/r1_deep_keyed_lm.py', 'experiments/r1_token_keyed_lm.py')})
    with path.open('x') as f: json.dump(out, f, indent=1); f.write('\n')
    print(json.dumps({k: out[k] for k in out if k in ('status', 'abs_diff', 'public_cost_estimate_h', 'public', 'max_rss_kb')}), flush=True)


if __name__ == '__main__':
    main()
