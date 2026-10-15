"""R1: do models learn document-level concepts? Loss by position in the document, with full vs 64-token history.

Documents: the first `--docs` end-of-text-delimited documents of at least 1024 tokens in fineweb_val after token
26,000,000 (disjoint from the selection, 65,528 and flagship slices; validation data is never trained on). Each lane is
[end-of-text, first 1024 document tokens], scored from fresh state. Per target: NLL, role (part_whole_structure roles)
and position in the document. Buckets [0,16), [16,64), [64,256), [256,1024).
Document-context value = NLL with history cut to the last 64 tokens minus full-history NLL, by bucket and role: the part
of the prediction that depends on having read the document beyond the local window, i.e. on document-level variables.
Models: r1_token_keyed_lm (keyed), r1_deep_keyed_lm / r1_deep_keyed_stream (plan-B), r1_part_whole_lm (hierarchy).
"""
import argparse, json, sys, time, hashlib
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'experiments'), str(ROOT / 'experiments/tpp'), str(ROOT / 'analysis')]
import r1_token_keyed_lm as keyed
from r1_deep_keyed_lm import DeepKeyedLM
import r1_part_whole_lm as pw

BUCKETS = ((0, 16), (16, 64), (64, 256), (256, 1024))
TABLE = pw.ROLES.numpy()
KIND = np.select([TABLE == 0, TABLE == 1, np.isin(TABLE, (2, 3, 4)), TABLE == 5], [0, 1, 2, 3], 4).astype(np.int8)
ROLE_NAMES = ('word_initial', 'continuation', 'punctuation', 'newline', 'other')


def documents(val, n, start=26_000_000, length=1024):
    data = np.asarray(val[start:start + 20_000_000]); eos = np.flatnonzero(data == 50256); out = []
    for a, b in zip(eos[:-1], eos[1:]):
        if b - a - 1 >= length: out.append(data[a:a + length + 1].astype(np.int64))   # [EOS, doc tokens...]
        if len(out) == n: break
    return np.stack(out), start


def build(run):
    rec = json.loads((ROOT / f'experiments/results/token_language/{run}.json').read_text()); g = rec['args']
    if 'horizons' in g: m = pw.PartWholeLM(g['d'], g['modes'], g['layers'], g['heads'], g['dk'], g['dropout'], tuple(g['horizons']))
    elif 'checkpoint_blocks' in g: m = DeepKeyedLM(g['d'], g['modes'], g['layers'], g['heads'], g['dk'], g['dropout'])
    else: m = keyed.KeyedTokenLM(g['d'], g['modes'], g['layers'], g['heads'], g['dk'], g['keyed'], g['dropout'])
    m.load_state_dict(torch.load(ROOT / rec['checkpoint'])); m.eval(); return m


@torch.no_grad()
def losses(model, lanes, cut=None):
    out = []
    for lane in lanes:
        x = torch.from_numpy(lane)[None]; h, e = model.states(x[:, :-1]); L = x.shape[1] - 1
        row = F.cross_entropy(model.logits(h, e, x[:, :-1], list(range(L)))[0], x[0, 1:], reduction='none')
        if cut is not None:
            for s in range(cut - 1, L, 128):
                idx = torch.arange(s, min(s + 128, L)); sub = torch.stack([x[0, t - cut + 1:t + 1] for t in idx.tolist()])
                hh, ee = model.states(sub); row[idx] = F.cross_entropy(model.logits(hh, ee, sub, [cut - 1])[:, 0], x[0, 1:][idx], reduction='none')
        out.append(row.double().numpy())
    return np.stack(out)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--tag', required=True); p.add_argument('--runs', nargs='+', required=True); p.add_argument('--docs', type=int, default=32); a = p.parse_args()
    torch.set_num_threads(1); began = time.monotonic(); val = keyed.load(ROOT / 'data/fineweb_gpt2/fineweb_val_000000.bin')
    lanes, start = documents(val, a.docs); inputs, labels = lanes[:, :-1], lanes[:, 1:]
    roles = KIND[labels].copy(); roles[(roles == 1) & (KIND[inputs] == 3)] = 0
    pos = np.broadcast_to(np.arange(labels.shape[1]), labels.shape); rows = {}
    for run in a.runs:
        m = build(run); full = losses(m, lanes); short = losses(m, lanes, cut=64); row = {}
        for lo, hi in BUCKETS:
            b = (pos >= lo) & (pos < hi); wi = b & (roles == 0)
            row[f'{lo}-{hi}'] = dict(nll=float(full[b].mean()), word_initial_nll=float(full[wi].mean()),
                                     document_context_value=float((short - full)[b].mean()), word_initial_document_context_value=float((short - full)[wi].mean()))
        rows[run] = row; print(run, {k: (round(v['word_initial_nll'], 3), round(v['word_initial_document_context_value'], 3)) for k, v in row.items()}, flush=True)
    out = dict(status='completed', battle='R1', tag=a.tag, runs=a.runs, rows=rows, protocol=dict(documents=a.docs, search_start=start, lane='[EOS] + first 1024 document tokens, fresh state', history_cut=64, buckets=BUCKETS),
               source_sha256={n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in ('analysis/document_concept_curve.py', 'experiments/r1_part_whole_lm.py', 'experiments/r1_deep_keyed_lm.py', 'experiments/r1_token_keyed_lm.py')},
               process_wall_s=time.monotonic() - began, scope='Frozen models; development data; single seeds.')
    with (ROOT / f'experiments/results/diagnostics/{a.tag}.json').open('x') as f: json.dump(out, f, indent=2); f.write('\n')


if __name__ == '__main__':
    main()
