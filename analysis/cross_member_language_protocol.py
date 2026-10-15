"""R1: score the keyed temporal-memory token members on the flagship's protocol.

Flagship protocol (public_token_scoring.windows): fineweb_val_000000, targets at
indices 25,165,825..25,174,016 (8192), disjoint 1024-token windows, each target
conditioning only on its window's history. The keyed members (r1_token_keyed_lm.py)
were selected and scored on their own 65,528-target slice with up to 8191-token lane
histories; this puts them in the same table as the 5.978388 flagship. Frozen
checkpoints, no fitting, no selection.
"""
import argparse, hashlib, json, resource, sys, time
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'experiments')]
import r1_token_keyed_lm as keyed

RUNS = ('curie_r1g2_keyed0_1M_s0_20261007T1800Z', 'curie_r1g2_keyed1_1M_s0_20261007T1800Z', 'curie_r1g2_keyed1_1M_s1_20261007T2055Z',
        'curie_r1g2_keyed1_1M_s2_20261007T2055Z', 'curie_r1g2_keyed1_4M_s0_20261008T1140Z', 'curie_r1g2_keyed1_4M_s1_20261008T1840Z')
START, TARGETS, WINDOW = 25165824, 8192, 1024


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(); p.add_argument('--tag', required=True); a = p.parse_args(); torch.set_num_threads(1); began = time.monotonic()
    val = keyed.load(ROOT / 'data/fineweb_gpt2/fineweb_val_000000.bin'); rows = []
    for run in RUNS:
        rec = json.loads((ROOT / f'experiments/results/token_language/{run}.json').read_text()); g = rec['args']
        model = keyed.KeyedTokenLM(g['d'], g['modes'], g['layers'], g['heads'], g['dk'], g['keyed'], g['dropout'])
        model.load_state_dict(torch.load(ROOT / rec['checkpoint'])); model.eval()
        total = n = 0
        with torch.no_grad():
            for begin in range(START, START + TARGETS, WINDOW):
                block = np.asarray(val[begin:begin + WINDOW + 1])[None]
                t_, n_ = keyed.lane_nll(model, block, 256); total += t_; n += n_
        assert n == TARGETS
        rows.append(dict(run=run, keyed=g['keyed'], train_tokens=g['train_tokens'], seed=g['seed'], best_epoch=rec['best_epoch'], parameters=rec['parameters'],
                         flagship_protocol_nll=total / n, own_slice_nll=rec['score_65528']['nll'], checkpoint_sha256=sha(ROOT / rec['checkpoint'])))
        print(json.dumps(rows[-1]), flush=True)
    out = dict(status='completed', battle='R1', tag=a.tag, rows=rows,
               flagship_reference=dict(run='aws_memory_policy_corpus_eos_2m_20261015_v1', nll=5.978387557528713),
               protocol=dict(file='fineweb_val_000000', target_start=START + 1, targets=TARGETS, window=WINDOW, reset='every window'),
               source_sha256={s: sha(ROOT / s) for s in ('analysis/cross_member_language_protocol.py', 'experiments/r1_token_keyed_lm.py', 'experiments/tpp/race_tpp_v5.py')},
               max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, process_wall_s=time.monotonic() - began,
               scope='Frozen keyed members scored once on the flagship 8192-target/1024-window protocol. Members differ in training data presentation (keyed: repeated epochs over 1M/4M; flagship: ~15M fresh targets), architecture and work; this is a quality placement, not a matched-compute comparison.')
    with (ROOT / f'experiments/results/diagnostics/{a.tag}.json').open('x') as f: json.dump(out, f, indent=2); f.write('\n')


if __name__ == '__main__':
    main()
