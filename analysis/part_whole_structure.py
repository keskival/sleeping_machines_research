"""R1: part-whole structure of the token stream and where prediction loss and context value live (no fitting).

GPT-2 byte-level BPE marks a word start by a leading space (U+0120 'G-dot' in the vocabulary) and newlines by U+010A.
Token roles: word_initial (space- or line-initial word), continuation (later piece of a word), punctuation, newline,
other. The analysis reports
1. structure of fineweb_val: tokens per word, words per sentence, sentences per paragraph, tokens per document;
2. per-role mean NLL and share of total loss on the flagship protocol (8192 targets, 1024-token windows) for the
   strongest native member (keyed 4M s0), race-stack ladder arms and a training-unigram reference;
3. context value by role for the keyed member: NLL with history cut to the last 16 / 64 tokens vs the full window;
4. per-stage unique value by role for race-stack arms (stage readout input mean-ablated, as depth_usage_probe).
Vocabulary: GPT-2 encoder.json (openaipublic 124M), hashed into the result.
"""
import argparse, json, sys, time, hashlib
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'experiments'), str(ROOT / 'analysis'), str(ROOT / 'experiments/tpp')]
import r1_token_keyed_lm as keyed
import depth_usage_probe as probe

START, TARGETS, WINDOW = 25165824, 8192, 1024
ROLES = ('word_initial', 'continuation', 'punctuation', 'newline', 'other')
PUNCT = set('.,;:!?"\'()[]{}-')


def vocab_roles(path):
    enc = json.loads(Path(path).read_text()); tok = [None] * len(enc)
    for s, i in enc.items(): tok[i] = s
    space, nl = 'Ġ', 'Ċ'
    kind = np.zeros(len(tok), dtype=np.int8); sentence_end = np.zeros(len(tok), dtype=bool)
    for i, s in enumerate(tok):
        core = s.lstrip(space)
        if nl in s: kind[i] = 3
        elif core and all(c in PUNCT for c in core): kind[i] = 2; sentence_end[i] = core.rstrip('"\')') in ('.', '!', '?', '...')
        elif s.startswith(space) and core[:1].isalnum(): kind[i] = 0
        elif core[:1].isalnum(): kind[i] = 1
        else: kind[i] = 4
    kind[50256] = 4
    return kind, sentence_end


def target_roles(inputs, labels, kind):
    r = kind[labels].copy()
    # A word piece at the start of a line (after a newline token, no leading space) starts a word.
    r[(r == 1) & (kind[inputs] == 3)] = 0
    return r


def structure(tokens, kind, sentence_end):
    k = kind[tokens]; prev_nl = np.r_[False, k[:-1] == 3]
    word_start = (k == 0) | ((k == 1) & prev_nl)
    words = int(word_start.sum()); pieces = int(((k == 0) | (k == 1)).sum())
    sentences = int(sentence_end[tokens].sum()); paragraphs = int(((k == 3)).sum()); docs = int((tokens == 50256).sum())
    return dict(tokens=len(tokens), word_pieces_per_word=pieces / words, tokens_per_word=len(tokens) / words, words_per_sentence=words / max(sentences, 1),
                sentences_per_newline=sentences / max(paragraphs, 1), tokens_per_document=len(tokens) / max(docs, 1),
                role_fraction={r: float((k == i).mean()) for i, r in enumerate(ROLES)})


def by_role(loss, roles):
    total = loss.sum()
    return {r: dict(fraction=float((roles == i).mean()), nll=float(loss[roles == i].mean()) if (roles == i).any() else None,
                    loss_share=float(loss[roles == i].sum() / total)) for i, r in enumerate(ROLES)} | dict(all=float(loss.mean()))


@torch.no_grad()
def keyed_losses(model, val, cut):
    out = []
    for begin in range(START, START + TARGETS, WINDOW):
        block = torch.from_numpy(np.asarray(val[begin:begin + WINDOW + 1]).astype(np.int64))
        if cut is None:
            x = block[None]; h, e = model.states(x[:, :-1]); lg = model.logits(h, e, x[:, :-1], list(range(WINDOW)))
            out.append(F.cross_entropy(lg[0], x[0, 1:], reduction='none'))
        else:
            # Target t conditions only on the last `cut` inputs; targets with fewer prior inputs keep their full-window loss.
            x = block[None]; h, e = model.states(x[:, :-1]); row = F.cross_entropy(model.logits(h, e, x[:, :-1], list(range(WINDOW)))[0], x[0, 1:], reduction='none')
            ids, tgt = block[:-1], block[1:]
            for s in range(cut - 1, WINDOW, 256):
                idx = torch.arange(s, min(s + 256, WINDOW))
                lanes = torch.stack([ids[t - cut + 1:t + 1] for t in idx.tolist()])
                h, e = model.states(lanes); lg = model.logits(h, e, lanes, [cut - 1])[:, 0]
                row[idx] = F.cross_entropy(lg, tgt[idx], reduction='none')
            out.append(row)
    return torch.cat(out).double().numpy()


def main():
    p = argparse.ArgumentParser(); p.add_argument('--tag', required=True); p.add_argument('--vocab', required=True)
    p.add_argument('--keyed-run', default='curie_r1g2_keyed1_4M_s0_20261008T1140Z')
    p.add_argument('--race-arms', nargs='*', default=['curie_depth_ladder_d2_residual_reads_1m_20261015_v1', 'curie_depth_ladder_d8_residual_reads_1m_20261015_v1'])
    a = p.parse_args(); torch.set_num_threads(1); began = time.monotonic()
    kind, sentence_end = vocab_roles(a.vocab); val = keyed.load(ROOT / 'data/fineweb_gpt2/fineweb_val_000000.bin')
    ids = np.asarray(val[START:START + TARGETS + 1]).astype(np.int64); inputs, labels = ids[:-1], ids[1:]
    roles = target_roles(inputs, labels, kind)
    out = dict(status='completed', battle='R1', tag=a.tag, structure=structure(np.asarray(val[:2_097_152]).astype(np.int64), kind, sentence_end), models={})
    # Unigram reference from the training counts used by the race arms.
    tr = keyed.load(ROOT / 'data/fineweb_gpt2/fineweb_train_000001.bin'); counts = np.bincount(np.asarray(tr[:4_194_304]), minlength=50257) + .1
    uni = -np.log(counts / counts.sum())[labels]; out['models']['unigram_4M_train'] = by_role(uni, roles)
    rec = json.loads((ROOT / f'experiments/results/token_language/{a.keyed_run}.json').read_text()); g = rec['args']
    m = keyed.KeyedTokenLM(g['d'], g['modes'], g['layers'], g['heads'], g['dk'], g['keyed'], g['dropout']); m.load_state_dict(torch.load(ROOT / rec['checkpoint'])); m.eval()
    full = keyed_losses(m, val, None); out['models'][a.keyed_run] = by_role(full, roles)
    ctx = {}
    for cut in (16, 64):
        short = keyed_losses(m, val, cut); ctx[str(cut)] = {r: float((short - full)[roles == i].mean()) for i, r in enumerate(ROLES)} | dict(all=float((short - full).mean()))
        print('keyed context value beyond', cut, {k: round(v, 3) for k, v in ctx[str(cut)].items()}, flush=True)
    out['keyed_context_value_by_role'] = ctx
    for run in a.race_arms:
        model, args = probe.load(run); dev = probe.driver.load_tokens(args.dev_file)
        x, y = probe.features(model, args, dev)
        # features() orders targets lane-major per chunk; rebuild per-target order.
        D = args.depth; Wd = x.shape[1] // D; chunk = args.chunk
        order = torch.arange(len(y)).reshape(WINDOW // chunk, 8, chunk).permute(1, 0, 2).reshape(-1)
        def loss_of(z):
            with torch.no_grad(): l = torch.cat([model.readout.nll(z[i:i + 1024], y[i:i + 1024]).double() for i in range(0, len(y), 1024)])
            return l[order].numpy()
        base = loss_of(x); assert np.array_equal(y[order].numpy(), labels)
        row = by_role(base, roles); mean = x.mean(0).reshape(D, Wd); stage = []
        for d in range(D):
            z = x.clone().reshape(-1, D, Wd); z[:, d] = mean[d]; dl = loss_of(z.reshape(len(y), -1)) - base
            stage.append({r: float(dl[roles == i].mean()) for i, r in enumerate(ROLES)})
        row['stage_unique_value_by_role'] = stage; out['models'][run] = row
        print(run, {r: round(row[r]['nll'], 3) for r in ROLES}, flush=True)
    out['source_sha256'] = {n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in ('analysis/part_whole_structure.py', 'experiments/r1_token_keyed_lm.py', 'analysis/depth_usage_probe.py')}
    out['vocab'] = dict(source='https://openaipublic.blob.core.windows.net/gpt-2/models/124M/encoder.json', sha256=hashlib.sha256(Path(a.vocab).read_bytes()).hexdigest())
    out['process_wall_s'] = time.monotonic() - began
    with (ROOT / f'experiments/results/diagnostics/{a.tag}.json').open('x') as f: json.dump(out, f, indent=2); f.write('\n')
    print(json.dumps(dict(structure=out['structure'], keyed={r: out['models'][a.keyed_run][r] for r in ROLES})))


if __name__ == '__main__':
    main()
