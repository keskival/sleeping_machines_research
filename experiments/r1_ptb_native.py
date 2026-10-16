#!/usr/bin/env python3
"""R1 published word-level benchmarks (Penn Treebank, WikiText-2): the deep keyed temporal-memory member trained on the
benchmark's own training split and scored on its standard protocol, frozen and with causal online weight adaptation.

Protocol (experiments/SCALING_SUPREMACY_PROTOCOL.md; references Melis et al. ICLR 2020 Table 1): Mikolov PTB split
(data/ptb/ptb.{train,valid,test}.txt, 10,000-word vocabulary incl. <unk>) and WikiText-2 (data/wikitext2/wiki.*.tokens,
33,278 words; --corpus wt2), each line's words followed by <eos> (Merity et al. AWD-LSTM convention), the
split read as one continuous stream. Every token of a split is a target (82,430 test, 73,760 valid), the first
conditioned on a start <eos>; perplexity = exp(mean NLL). History crosses sentence boundaries, as in the published
recurrent and Transformer-XL evaluations. The test split is sealed: development runs score validation only; `--mode
score --split test` scores a selected checkpoint once and refuses to overwrite.

Model (r1_deep_keyed_lm.py blocks unchanged): token embedding tied with the output, pre-norm residual blocks of exact
parallel temporal memory (complex-diagonal decay/rotation per step), per-layer keyed race read over past events (keys and
values from each event's state and its predecessor's message), MLP; output keyed token-slot bonus (each past event writes
a key to its own word's slot, the query adds decayed slot scores to those words' logits: a learned cache pointer trained
end to end). Small-corpus regularisation (standard primitives): word-type embedding dropout, input/residual/output
dropout, decoupled weight decay.

Training: each epoch cuts the training stream at a fresh random offset into lanes of --lane+1 tokens (every target
conditions on its lane history), shuffled, AdamW with warmup + cosine decay; validation each epoch; the best checkpoint
by validation is kept. Exact resume at epoch boundaries (model, optimiser, schedule, RNG, history).

Scoring arms (both causal; every target scored once):
- frozen: sliding windows of --ctx tokens advanced by --stride; each window scores only its last `stride` targets, so
  every target sees at least ctx-stride tokens of history (the persistent context across sentences and articles).
- online: the same windows, but after scoring a segment the weights take one step on that segment's loss (dynamic
  evaluation, Krause et al. 2018: gradient normalised by training-set RMS, decay toward the trained weights). A segment's
  predictions use weights adapted only on earlier segments (contract: perturbing token p leaves every score before p
  unchanged). Hyper-parameters (lr, decay) are chosen on validation only.
"""
import argparse, hashlib, json, math, os, resource, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[1]; sys.path[:0] = [str(ROOT/'experiments'), str(ROOT/'experiments/tpp')]
from r1_deep_keyed_lm import Block  # noqa: E402
from r1_token_keyed_lm import KeyedTokenLM  # noqa: E402

CORPORA = dict(ptb=dict(files={s: ROOT/f'data/ptb/ptb.{s}.txt' for s in ('train', 'valid', 'test')}, vocab=10000,
                        counts=dict(train=929589, valid=73760, test=82430)),
               wt2=dict(files={s: ROOT/f'data/wikitext2/wiki.{s}.tokens' for s in ('train', 'valid', 'test')}, vocab=33278,   # scripts/convert_wikitext2_parquet.py
                        counts=dict(train=2088628, valid=217646, test=245569)))
EXECUTION_ONLY = ('max_steps', 'max_wall_s', 'mode', 'run', 'split', 'select_from', 'online_lr', 'online_decay', 'online_stride', 'ms_batches', 'tm_chunk', 'compile_blocks', 'grid_targets')


def read_corpus(name):
    c = CORPORA[name]; words = {s: [w for line in open(p) for w in line.split() + ['<eos>']] for s, p in c['files'].items()}
    vocab = sorted(set(words['train'])); assert len(vocab) == c['vocab'], len(vocab)
    idx = {w: i for i, w in enumerate(vocab)}; ids = {s: np.array([idx[w] for w in ws], dtype=np.int64) for s, ws in words.items()}
    for s, n in c['counts'].items(): assert len(ids[s]) == n, (s, len(ids[s]))
    return ids, vocab, idx['<eos>']


class NativeWordLM(nn.Module):
    """DeepKeyedLM generalised to any vocabulary, with small-corpus dropout."""
    keyed_scores, logits = KeyedTokenLM.keyed_scores, KeyedTokenLM.logits
    train_blocks = None                                                 # torch.compile'd views of self.blocks (training only)

    def __init__(self, vocab, d, modes, layers, heads, dk, dropout, drop_emb, drop_in, drop_out, out_heads=4, out_dk=16):
        super().__init__(); self.keyed, self.heads, self.dk = 1, out_heads, out_dk
        self.drop_emb, self.drop_in, self.drop_out = drop_emb, nn.Dropout(drop_in), nn.Dropout(drop_out)
        self.embed = nn.Embedding(vocab, d); nn.init.normal_(self.embed.weight, std=0.02)
        self.blocks = nn.ModuleList(Block(d, modes, heads, dk, dropout, layers) for _ in range(layers))
        self.out_norm = nn.LayerNorm(d); self.bias = nn.Parameter(torch.zeros(vocab))
        self.key_w = nn.Linear(2*d, out_heads*out_dk); self.query_w = nn.Linear(d, out_heads*out_dk)
        self.log_rate = nn.Parameter(torch.linspace(math.log(1e-3), math.log(0.3), out_heads))

    def states(self, x):
        w = self.embed.weight
        if self.training and self.drop_emb > 0:                     # drop whole word types for this batch (rescaled)
            keep = torch.bernoulli(torch.full((w.shape[0], 1), 1 - self.drop_emb)); w = w*keep/(1 - self.drop_emb)
        e = self.drop_in(F.embedding(x, w)); h = e
        for b in (self.train_blocks if (self.training and self.train_blocks) else self.blocks): h = b(h, e)
        return self.drop_out(self.out_norm(h)), e


def window_segments(model, stream, ctx, stride, start_token):
    """Yield per-target NLL tensors for consecutive segments of `stride` targets covering every token of `stream` (the first
    conditioned on start_token); each segment's window holds up to ctx tokens of preceding history."""
    x = np.concatenate([[start_token], stream]); n = len(stream)
    for t0 in range(0, n, stride):
        t1 = min(t0 + stride, n); w0 = max(0, t1 - ctx)                 # targets t0..t1-1 = positions t0+1..t1 of x
        inp = torch.from_numpy(x[w0:t1][None]); h, e = model.states(inp)
        yield F.cross_entropy(model.logits(h, e, inp, list(range(t0 - w0, t1 - w0)))[0], torch.from_numpy(x[t0 + 1:t1 + 1]), reduction='none')


def frozen_score(model, stream, ctx, stride, start_token):
    model.eval()
    with torch.no_grad(): return torch.cat(list(window_segments(model, stream, ctx, stride, start_token))).numpy()


def training_ms(model, train, lane, batches, seed):
    """Mean squared gradient over training lanes (Krause et al. RMS normaliser; training data only)."""
    model.eval(); ms = [torch.zeros_like(p) for p in model.parameters()]; rng = np.random.default_rng(seed)
    for _ in range(batches):
        i = rng.integers(0, len(train) - lane - 1); x = torch.from_numpy(train[i:i + lane + 1][None]); h, e = model.states(x[:, :-1])
        loss = F.cross_entropy(model.logits(h, e, x[:, :-1], list(range(lane)))[0], x[0, 1:]); model.zero_grad(); loss.backward()
        for m, p in zip(ms, model.parameters()): m += p.grad**2/batches
    return ms


def online_score(model, stream, ctx, stride, start_token, ms, lr, decay):
    """Dynamic evaluation: score a segment, then theta <- theta - lr*g/(rms+eps) + decay*(rms/mean rms)*(theta0 - theta)."""
    model.eval(); params = list(model.parameters()); theta0 = [p.detach().clone() for p in params]
    rms = []
    for m in ms:                                                        # floor: entries unseen in the training sample (rare-word rows) use half the tensor's typical RMS
        r = m.sqrt(); nz = r[r > 0]; rms.append(r.clamp_min(0.5*nz.mean().item() if nz.numel() else 1e-3) + 1e-8)
    mean_rms = torch.cat([r.flatten() for r in rms]).mean(); dec = [(decay*r/mean_rms).clamp(max=1.0) for r in rms]
    out = []
    for nll in window_segments(model, stream, ctx, stride, start_token):
        out.append(nll.detach().numpy()); model.zero_grad(); nll.mean().backward()
        with torch.no_grad():
            for p, r, d_, t0 in zip(params, rms, dec, theta0): p.add_(-lr*p.grad/r + d_*(t0 - p))
    with torch.no_grad():
        for p, t0 in zip(params, theta0): p.copy_(t0)
    return np.concatenate(out)


def contracts(a, ids, eos):
    """Causality of both arms, exact target coverage and weight restoration, on a small random model (seconds)."""
    V = CORPORA[a.corpus]['vocab']; torch.manual_seed(0); m = NativeWordLM(V, 32, 8, 2, 2, 8, 0.0, 0.0, 0.0, 0.0); w0 = [q.detach().clone() for q in m.parameters()]
    s = ids['valid'][:300].copy(); s2 = s.copy(); p = 170; s2[p] = (s2[p] + 1) % V; res = {}
    f1, f2 = frozen_score(m, s, 96, 32, eos), frozen_score(m, s2, 96, 32, eos)
    res['frozen_targets'] = len(f1); res['frozen_prefix_max_diff'] = float(np.abs(f1[:p] - f2[:p]).max()); res['frozen_changed_at_p'] = bool(abs(f1[p] - f2[p]) > 0)
    full = frozen_score(m, s, 400, 400, eos); res['frozen_vs_full_context_first_window'] = float(np.abs(full[:96] - f1[:96]).max())   # first ctx targets share their full history
    ms = [torch.full_like(q, 1e-4) for q in m.parameters()]
    o1, o2 = online_score(m, s, 96, 32, eos, ms, 1e-3, 0.01), online_score(m, s2, 96, 32, eos, ms, 1e-3, 0.01)
    res['online_targets'] = len(o1); res['online_prefix_max_diff'] = float(np.abs(o1[:p] - o2[:p]).max()); res['online_changed_after_p'] = float(np.abs(o1[p:] - o2[p:]).max())
    res['online_first_segment_equals_frozen'] = float(np.abs(o1[:32] - f1[:32]).max()); res['online_adapts'] = float(np.abs(o1[32:] - f1[32:]).max())
    res['weights_restored'] = all(torch.equal(q, r) for q, r in zip(m.parameters(), w0))
    ok = (res['frozen_targets'] == res['online_targets'] == len(s) and res['frozen_prefix_max_diff'] == 0 and res['frozen_changed_at_p'] and res['frozen_vs_full_context_first_window'] < 1e-5
          and res['online_prefix_max_diff'] == 0 and res['online_changed_after_p'] > 0 and res['online_first_segment_equals_frozen'] < 1e-5 and res['online_adapts'] > 0 and res['weights_restored'])
    res['status'] = 'passed' if ok else 'failed'; return res


def build(a):
    m = NativeWordLM(CORPORA[a.corpus]['vocab'], a.d, a.modes, a.layers, a.heads, a.dk, a.dropout, a.drop_emb, a.drop_in, a.drop_out)
    for b in m.blocks: b.tm.CHUNK = getattr(a, 'tm_chunk', 256)       # exact for any chunk (state carried across chunks); execution only
    if getattr(a, 'compile_blocks', 0): m.train_blocks = [torch.compile(b, dynamic=False) for b in m.blocks]   # plain list: shares parameters, state_dict unchanged
    return m


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True); ap.add_argument('--mode', choices=('train', 'score', 'contracts'), default='train')
    ap.add_argument('--corpus', default='ptb'); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--d', type=int, default=256); ap.add_argument('--modes', type=int, default=64); ap.add_argument('--layers', type=int, default=8)
    ap.add_argument('--heads', type=int, default=4); ap.add_argument('--dk', type=int, default=32)
    ap.add_argument('--dropout', type=float, default=0.3); ap.add_argument('--drop-emb', type=float, default=0.1); ap.add_argument('--drop-in', type=float, default=0.4)
    ap.add_argument('--drop-out', type=float, default=0.4); ap.add_argument('--lr', type=float, default=2e-3); ap.add_argument('--wd', type=float, default=0.1)
    ap.add_argument('--lane', type=int, default=512); ap.add_argument('--batch', type=int, default=4); ap.add_argument('--epochs', type=int, default=16)
    ap.add_argument('--warmup', type=float, default=0.02); ap.add_argument('--ctx', type=int, default=512); ap.add_argument('--stride', type=int, default=128)
    ap.add_argument('--max-steps', type=int, default=0); ap.add_argument('--max-wall-s', type=float, default=0, help='stop at an epoch boundary after this wall time (resumable)')
    ap.add_argument('--run', default=''); ap.add_argument('--split', default='valid'); ap.add_argument('--select-from', default='', help='test: the validation online-score tag whose selected (lr, decay) is used')
    ap.add_argument('--online-lr', default='3e-5,1e-4,3e-4'); ap.add_argument('--online-decay', default='0,0.002'); ap.add_argument('--online-stride', type=int, default=64)
    ap.add_argument('--grid-targets', type=int, default=16384, help='validation: choose (lr, decay) on the first N validation targets, then score the full split once with the choice')
    ap.add_argument('--ms-batches', type=int, default=64)
    ap.add_argument('--compile-blocks', type=int, default=0, help='train through torch.compile\'d blocks (execution only; scoring runs eager)')
    ap.add_argument('--tm-chunk', type=int, default=64, help='temporal-memory kernel chunk (exact for any value; execution only)')
    a = ap.parse_args(); torch.set_num_threads(1); torch.manual_seed(a.seed); began = time.time()
    ids, vocab, eos = read_corpus(a.corpus); out_dir = ROOT/'experiments/results/word_language'; out_dir.mkdir(exist_ok=True)
    src = {n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ('experiments/r1_ptb_native.py', 'experiments/r1_deep_keyed_lm.py', 'experiments/r1_token_keyed_lm.py')}
    if a.mode == 'contracts':
        res = dict(battle='R1', tag=a.tag, corpus=a.corpus, **contracts(a, ids, eos), source_sha256=src)
        with (ROOT/f'experiments/results/diagnostics/{a.tag}.json').open('x') as f: json.dump(res, f, indent=1); f.write('\n')
        print(json.dumps(res)); return
    if a.mode == 'score':
        return score(a, ids, eos, out_dir, src, began)
    out = out_dir/f'{a.tag}.json'; state_path = out_dir/f'{a.tag}.resume.pt'; ckpt = out_dir/f'{a.tag}.pt'; assert not out.exists()
    model = build(a); params = sum(p.numel() for p in model.parameters()); train = ids['train']
    steps_per_epoch = (len(train) - 1)//(a.lane*a.batch); total = steps_per_epoch*a.epochs; warm = max(1, int(a.warmup*total))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd, betas=(0.9, 0.98))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1)/warm)*0.5*(1 + math.cos(math.pi*min(1.0, s/total))))
    ident = {k: v for k, v in vars(a).items() if k not in EXECUTION_ONLY}
    epoch, step, best, best_epoch, history, wall, resumes = 0, 0, math.inf, -1, [], 0.0, 0; rng = np.random.default_rng(a.seed)
    if state_path.exists():
        s = torch.load(state_path, weights_only=False); assert s['args'] == ident, 'resume with different settings'
        model.load_state_dict(s['model']); opt.load_state_dict(s['opt']); sched.load_state_dict(s['sched']); torch.set_rng_state(s['torch_rng']); rng.bit_generator.state = s['np_rng']
        epoch, step, best, best_epoch, history, wall, resumes = s['epoch'], s['step'], s['best'], s['best_epoch'], s['history'], s['wall'], s['resumes'] + 1
        print(json.dumps(dict(resumed_at_epoch=epoch)), flush=True)
    t_start = time.time() - wall
    while epoch < a.epochs:
        model.train(); e0 = time.time(); off = int(rng.integers(0, a.lane)); n_l = (len(train) - 1 - off)//a.lane
        lanes = np.stack([train[off + i*a.lane: off + (i + 1)*a.lane + 1] for i in range(n_l)]); perm = rng.permutation(n_l); tl = tn = 0.0
        for bi in range(steps_per_epoch):
            x = torch.from_numpy(lanes[perm[bi*a.batch:(bi + 1)*a.batch]]); h, e = model.states(x[:, :-1])
            loss = F.cross_entropy(model.logits(h, e, x[:, :-1], list(range(a.lane))).reshape(-1, model.bias.shape[0]), x[:, 1:].reshape(-1))
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 0.5); opt.step(); sched.step(); step += 1
            tl += loss.item(); tn += 1
            if bi % 50 == 0: print(json.dumps(dict(epoch=epoch, step=bi, of=steps_per_epoch, train_nll=tl/tn, elapsed_s=round(time.time() - t_start))), flush=True)
            if a.max_steps and step >= a.max_steps:
                print(json.dumps(dict(smoke_steps=step, s_per_step=(time.time() - e0)/(bi + 1), tokens_per_s=(bi + 1)*a.batch*a.lane/(time.time() - e0),
                                      max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, parameters=params)), flush=True)
                v0 = time.time(); vn = frozen_score(model, ids['valid'][:4096], a.ctx, a.stride, eos)
                print(json.dumps(dict(smoke_valid4096_nll=float(vn.mean()), valid_s_per_1k_targets=(time.time() - v0)/4.096, max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)), flush=True); return
        v0 = time.time(); vn = float(frozen_score(model, ids['valid'], a.ctx, a.stride, eos).mean())
        history.append(dict(epoch=epoch, train_nll=tl/tn, valid_nll=vn, valid_ppl=math.exp(vn), epoch_s=round(time.time() - e0), valid_s=round(time.time() - v0)))
        print(json.dumps(history[-1]), flush=True); epoch += 1
        if vn < best: best, best_epoch = vn, epoch - 1; torch.save(model.state_dict(), ckpt)
        tmp = state_path.with_suffix('.tmp'); torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), torch_rng=torch.get_rng_state(), np_rng=rng.bit_generator.state,
                   epoch=epoch, step=step, best=best, best_epoch=best_epoch, history=history, wall=time.time() - t_start, resumes=resumes, args=ident), tmp); os.replace(tmp, state_path)
        if a.max_wall_s and time.time() - t_start > a.max_wall_s and epoch < a.epochs:
            print(json.dumps(dict(paused_at_epoch=epoch, resumable=str(state_path.relative_to(ROOT)))), flush=True); sys.exit(75)
    result = dict(status='completed', battle='R1', stage=f'published word-level benchmark: {a.corpus} native development fit', tag=a.tag, args=vars(a), parameters=params,
                  best_epoch=best_epoch, best_valid_nll=best, best_valid_ppl=math.exp(best), epochs_run=epoch, steps=step, presentations=step*a.batch*a.lane,
                  train_wall_s=time.time() - t_start, resumes=resumes, history=history, checkpoint=str(ckpt.relative_to(ROOT)), max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  test_scored=False, source_sha256=src)
    out.write_text(json.dumps(result, indent=1) + '\n'); state_path.unlink()
    print('RESULT', json.dumps({k: result[k] for k in ('best_valid_ppl', 'best_epoch', 'parameters', 'train_wall_s')}), flush=True)


def score(a, ids, eos, out_dir, src, began):
    """Frozen and online arms of a trained checkpoint on one split; online hyper-parameters are chosen on validation."""
    rec = json.loads((out_dir/f'{a.run}.json').read_text()); g = argparse.Namespace(**rec['args']); model = build(g); model.load_state_dict(torch.load(ROOT/rec['checkpoint']))
    path = out_dir/f'{a.tag}.json'; assert not path.exists()
    if a.split == 'test':
        sel = json.loads((out_dir/f'{a.select_from}.json').read_text()); assert sel['split'] == 'valid' and sel['run'] == a.run and sel['online']['stride'] == a.online_stride and sel['ctx'] == a.ctx; grid = [(sel['selected']['lr'], sel['selected']['decay'])]
    else:
        grid = [(float(l), float(d)) for l in a.online_lr.split(',') for d in a.online_decay.split(',')]
    res = dict(battle='R1', tag=a.tag, run=a.run, corpus=g.corpus, split=a.split, targets=len(ids[a.split]), parameters=rec['parameters'], ctx=a.ctx, stride=a.stride)
    fz = frozen_score(model, ids[a.split], a.ctx, a.stride, eos); res['frozen'] = dict(nll=float(fz.mean()), ppl=math.exp(fz.mean()))
    print(json.dumps(res['frozen']), flush=True)
    ms = training_ms(model, ids['train'], g.lane, a.ms_batches, a.seed); res['online_grid'] = []
    if len(grid) > 1:                                                   # validation only: grid on a prefix, then the full split with the choice
        for lr, dec in grid:
            o = online_score(model, ids[a.split][:a.grid_targets], a.ctx, a.online_stride, eos, ms, lr, dec)
            res['online_grid'].append(dict(lr=lr, decay=dec, targets=a.grid_targets, nll=float(o.mean()), ppl=math.exp(o.mean()))); print(json.dumps(res['online_grid'][-1]), flush=True)
        best_cfg = min(res['online_grid'], key=lambda r: r['nll'] if math.isfinite(r['nll']) else math.inf)   # a diverged setting is never selected; grid = [(best_cfg['lr'], best_cfg['decay'])]
    o = online_score(model, ids[a.split], a.ctx, a.online_stride, eos, ms, *grid[0])
    res['online'] = dict(lr=grid[0][0], decay=grid[0][1], stride=a.online_stride, nll=float(o.mean()), ppl=math.exp(o.mean())); res['selected'] = dict(lr=grid[0][0], decay=grid[0][1])
    res.update(status='completed', wall_s=time.time() - began, max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, source_sha256=src)
    with path.open('x') as f: json.dump(res, f, indent=1); f.write('\n')
    print('RESULT', json.dumps(dict(frozen=res['frozen'], online=res['online'])), flush=True)


if __name__ == '__main__':
    main()
