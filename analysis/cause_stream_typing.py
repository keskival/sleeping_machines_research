"""R1 / theory 170: characterise the attributed-cause stream of a trained CERL member before composing on it.

Founder (16 Oct): "Typing is not that trivial … are they sparser in cadence? Do they represent similar things? In
Transformers every layer inputs and outputs the exact same number of tokens … we can't assume as much."

On the 65,528-target development slice (8 lanes x 8191 targets of fineweb_val_000000), for the posterior attribution of
every observed token (no gradient), this measures:
- CADENCE: fraction of tokens that start a new cause event (rho) and event durations, for several cause identities:
  raw symbol (unit id / bg / repeat / continue), cause type only, concept families (unit // 16, // 64), span-aware
  continuation (a continuation stays one event while its pointer advances j -> j+1), and attribution with hysteresis.
- MEANING: per concept unit, the entropy of the tokens it explains against the unigram entropy, the token-role mix, and its
  most frequent tokens (decoded); continuation span lengths; how often the winning cause is confident (posterior margin).
- INFORMATION: event-symbol entropy, its conditional entropy given the previous event (bigram fitted on the first half of
  the slice, scored on the second), bits per token that the event stream carries upward, and the share of tokens whose
  cause is predictable from the previous event.
"""
import argparse, collections, json, math, sys
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path[:0] = [str(ROOT/'experiments'), str(ROOT/'experiments/tpp')]
import r1_cerl as C  # noqa: E402
from r1_gpu_ladder import vocab_roles  # noqa: E402


def attribute_lane(m, w, chunk=512):
    """Per-token attribution: symbol, cause type, continuation pointer j, winner posterior and margin to the runner-up."""
    x = torch.from_numpy(w[None, :-1].astype(np.int64)); y = torch.from_numpy(w[None, 1:].astype(np.int64)); L = x.shape[1]; h = m.states(x); U = m.U
    sym, typ, ptr, conf, marg = [], [], [], [], []
    for r0 in range(0, L, chunk):
        r1 = min(r0 + chunk, L); _, parts, s_sel, sel, *_ = m.race(x, y, h, r0, r1, False); post = torch.softmax(parts, -1)[0]; K = s_sel.shape[-1]
        rep, cont = post[:, 1 + K:1 + K + r1], post[:, 1 + K + r1:]
        g = torch.cat([post[:, :1], post[:, 1:1 + K], rep.sum(-1, keepdim=True), cont.sum(-1, keepdim=True)], -1)
        s = torch.cat([torch.full((r1 - r0, 1), U), sel[0], torch.full((r1 - r0, 1), U + 1), torch.full((r1 - r0, 1), U + 2)], -1)
        top2 = g.topk(2, -1); i = top2.indices[:, 0]
        sym.append(s.gather(-1, i[:, None])[:, 0]); typ.append(torch.where(i == 0, 0, torch.where(i <= K, 1, torch.where(i == K + 1, 2, 3))))
        ptr.append(torch.where(i == K + 2, cont.argmax(-1), torch.where(i == K + 1, rep.argmax(-1), torch.full_like(i, -1))))
        conf.append(top2.values[:, 0]); marg.append(top2.values[:, 0] - top2.values[:, 1])
    return [torch.cat(v).numpy() for v in (sym, typ, ptr, conf, marg)]


def events(ids):
    ch = np.ones(len(ids), bool); ch[1:] = ids[1:] != ids[:-1]; return ch


def durations(ch):
    starts = np.flatnonzero(ch); return np.diff(np.append(starts, len(ch)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True); ap.add_argument('--run', required=True); ap.add_argument('--vocab', default=str(ROOT/'data/gpt2/encoder.json')); a = ap.parse_args()
    torch.set_num_threads(1); rc = json.loads((ROOT/f'experiments/results/token_language/{a.run}.json').read_text()); g = rc['args']
    tr = C.load_fineweb(ROOT/'data/fineweb_gpt2/fineweb_train_000001.bin'); n = g['train_tokens']//g['lane']*g['lane']
    counts = np.bincount(np.asarray(tr[:n]), minlength=50257).astype(float); cls, words = C.word_classes(counts, g['classes'])
    m = C.CERL(50257, cls, words, g['d'], g['modes'], g['layers'], g['dk'], g['units'], g['topk'], g['explore'], age_slope=g.get('age_slope', 1.5), product_keys=bool(g.get('product_keys', 0)))
    m.load_state_dict(torch.load(ROOT/rc['checkpoint'])); m.eval(); U = m.U
    enc = json.loads(Path(a.vocab).read_text()); dec = {i: s for s, i in enc.items()}; kind = vocab_roles(a.vocab)
    val = C.load_fineweb(ROOT/'data/fineweb_gpt2/fineweb_val_000000.bin'); lanes = [np.asarray(val[p:p + 8192]) for p in range(20971520, 20971520 + 8*8192, 8192)]
    A = collections.defaultdict(list); tgt_all = []
    with torch.no_grad():
        for w in lanes:
            s, t, p, c, mg = attribute_lane(m, w)
            for k, v in zip(('sym', 'typ', 'ptr', 'conf', 'marg'), (s, t, p, c, mg)): A[k].append(v)
            tgt_all.append(w[1:])
    res = dict(battle='R1', tag=a.tag, run=a.run, targets=int(sum(len(v) for v in A['sym'])))
    # ---- cadence under several cause identities
    def ident(kind_):
        out = []
        for s, t, p in zip(A['sym'], A['typ'], A['ptr']):
            if kind_ == 'symbol': out.append(s)
            elif kind_ == 'type': out.append(t)
            elif kind_.startswith('family'):
                f = int(kind_[6:]); out.append(np.where(s < U, s//f, U + (s - U)))
            elif kind_ == 'span_aware':                                                    # continuation stays one event while its pointer advances
                ids = s.copy().astype(np.int64); span = 0
                for i in range(1, len(s)):
                    if t[i] == 3 and t[i - 1] == 3 and p[i] == p[i - 1] + 1: ids[i] = ids[i - 1]
                    elif t[i] == 3: span += 1; ids[i] = 10**6 + span
                out.append(ids)
        return out
    cad = {}
    for k in ('symbol', 'type', 'family16', 'family64', 'span_aware'):
        ch = np.concatenate([events(v) for v in ident(k)]); du = np.concatenate([durations(events(v)) for v in ident(k)])
        cad[k] = dict(rho=float(ch.mean()), mean_duration=float(du.mean()), median_duration=float(np.median(du)), p90_duration=float(np.percentile(du, 90)), longest=int(du.max()))
    # hysteresis: keep the previous symbol unless the new winner is confident by a margin
    for tau in (0.2, 0.4):
        ids_all = []
        for s, mg in zip(A['sym'], A['marg']):
            ids = s.copy()
            for i in range(1, len(s)):
                if mg[i] < tau: ids[i] = ids[i - 1]
            ids_all.append(ids)
        ch = np.concatenate([events(v) for v in ids_all]); cad[f'hysteresis_margin{tau}'] = dict(rho=float(ch.mean()))
    res['cadence'] = cad
    # ---- meaning
    sym = np.concatenate(A['sym']); typ = np.concatenate(A['typ']); tg = np.concatenate(tgt_all); conf = np.concatenate(A['conf']); mg = np.concatenate(A['marg'])
    uni = np.bincount(tg, minlength=50257).astype(float); pu = uni/uni.sum(); H_uni = float(-(pu[pu > 0]*np.log2(pu[pu > 0])).sum())
    units = collections.Counter(sym[typ == 1].tolist()); rows = []
    for u, cnt in units.most_common():
        if cnt < 50: break
        tk = tg[(typ == 1) & (sym == u)]; c = np.bincount(tk); q = c[c > 0]/c.sum(); r = np.bincount(kind[tk], minlength=5)/len(tk)
        rows.append(dict(unit=int(u), targets=int(cnt), token_entropy_bits=float(-(q*np.log2(q)).sum()), distinct_tokens=int((c > 0).sum()),
                         roles=dict(zip(('word_initial', 'continuation', 'punct', 'newline', 'other'), [round(float(v), 3) for v in r])),
                         top_tokens=[dec.get(int(i), '?').replace('Ġ', ' ').replace('Ċ', '\\n') for i in np.argsort(-c)[:8]]))
    res['meaning'] = dict(unigram_entropy_bits=H_uni, concept_units_used=len(units), units_with_50plus=len(rows), type_share={n_: float((typ == i).mean()) for i, n_ in enumerate(('background', 'concept', 'repeat', 'continue'))},
                          mean_winner_posterior=float(conf.mean()), share_margin_below_0_2=float((mg < 0.2).mean()),
                          concept_token_entropy_bits_weighted=float(sum(r['token_entropy_bits']*r['targets'] for r in rows)/max(1, sum(r['targets'] for r in rows))), top_units=rows[:25])
    spans = []
    for t, p in zip(A['typ'], A['ptr']):
        run = 0
        for i in range(len(t)):
            if t[i] == 3 and i > 0 and t[i - 1] == 3 and p[i] == p[i - 1] + 1: run += 1
            else:
                if run: spans.append(run + 1)
                run = 1 if t[i] == 3 else 0
    spans = np.array(spans or [0]); res['meaning']['copy_spans'] = dict(count=int(len(spans)), mean=float(spans.mean()), median=float(np.median(spans)), p90=float(np.percentile(spans, 90)), longest=int(spans.max()))
    # ---- information carried upward
    info = {}
    for k in ('symbol', 'family64', 'span_aware'):
        seqs = [v[events(v)] for v in ident(k)]; half = len(seqs)//2; fit, ev = seqs[:half], seqs[half:]
        vocab = {x: i for i, x in enumerate(sorted(set(np.concatenate(seqs).tolist())))}; Vn = len(vocab)
        cu = collections.Counter(); cb = collections.Counter(); cp = collections.Counter()
        for sq in fit:
            for i, x in enumerate(sq):
                cu[x] += 1
                if i: cb[(sq[i - 1], x)] += 1; cp[sq[i - 1]] += 1
        Nu = sum(cu.values()); hu = hb = 0.0; nn = 0; hit = 0
        for sq in ev:
            for i, x in enumerate(sq):
                pu_ = (cu[x] + 0.1)/(Nu + 0.1*Vn); hu -= math.log2(pu_)
                if i:
                    pb = (cb[(sq[i - 1], x)] + 0.1*Vn*pu_)/(cp[sq[i - 1]] + 0.1*Vn); hb -= math.log2(pb)
                    best = max(((cb[(sq[i - 1], y)], y) for y in cu if cb[(sq[i - 1], y)] > 0), default=(0, None))[1]; hit += best == x
                nn += 1
        tokens_per_event = 1.0/res['cadence'][k]['rho']
        info[k] = dict(alphabet=Vn, unigram_bits_per_event=hu/nn, bigram_bits_per_event=hb/max(1, nn - len(ev)), bits_per_token_upward=(hu/nn)/tokens_per_event,
                       next_event_top1_from_previous=hit/max(1, nn - len(ev)))
    res['information'] = info; res['status'] = 'completed'
    with (ROOT/f'experiments/results/diagnostics/{a.tag}.json').open('x') as f: json.dump(res, f, indent=1); f.write('\n')
    print(json.dumps(dict(cadence=cad, meaning={k: v for k, v in res['meaning'].items() if k != 'top_units'}, information=info), indent=1))
    for r in rows[:12]: print(r['unit'], r['targets'], round(r['token_entropy_bits'], 2), r['roles'], r['top_tokens'])


if __name__ == '__main__':
    main()
