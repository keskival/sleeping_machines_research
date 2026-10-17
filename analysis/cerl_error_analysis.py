"""R1 error analysis: native causal event race member (CERL) against the dense plan-B diagnostic, target by target.

On the flagship windows (8 x 1024 targets from 25,165,824) and the 65,528-target slice (8 lanes x 8191 from 20,971,520) of
fineweb_val_000000 this reports, for both models: NLL by token role (word-initial / continuation / punctuation / newline /
other, GPT-2 vocabulary), by history bucket (target offset in its window), by whether the target occurred earlier in the
window (copyable vs novel), and for CERL by winning cause type (posterior argmax: background / concept / repeat / continue)
with each cause type's share of targets and its NLL gap to the dense model. This locates where the race wins and which cause
must improve.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[1]; sys.path[:0] = [str(ROOT/'experiments'), str(ROOT/'experiments/tpp'), str(ROOT/'analysis')]
import r1_cerl as C  # noqa: E402
from r1_deep_keyed_lm import DeepKeyedLM  # noqa: E402
from r1_gpu_ladder import vocab_roles  # noqa: E402

ROLES = ('word_initial', 'continuation', 'punctuation', 'newline', 'other')


def cerl_rows(m, w):
    """Per-target NLL and winning cause type (0 bg, 1 concept, 2 repeat, 3 continue) for one lane w [L+1]."""
    x = torch.from_numpy(w[None, :-1].astype(np.int64)); y = torch.from_numpy(w[None, 1:].astype(np.int64)); L = x.shape[1]; nll, kind = [], []   # same x object for every chunk (RMR memory cache)
    for r0 in range(0, L, 512):
        r1 = min(r0 + 512, L); out = m.lane_nll(x, y, (r0, r1), stats=False); nll.append(out[0])
        kind.append(m.last_winner[0])
    return torch.cat(nll).numpy(), torch.cat(kind).numpy()


def dense_rows(m, w):
    x = torch.from_numpy(w[None, :-1].astype(np.int64)); y = torch.from_numpy(w[None, 1:].astype(np.int64)); h, e = m.states(x); out = []
    for r0 in range(0, x.shape[1], 256):
        rows = list(range(r0, min(r0 + 256, x.shape[1]))); out.append(F.cross_entropy(m.logits(h, e, x, rows)[0], y[0, rows], reduction='none'))
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True); ap.add_argument('--cerl', required=True); ap.add_argument('--dense', required=True)
    ap.add_argument('--vocab', default=str(ROOT/'data/gpt2/encoder.json')); a = ap.parse_args(); torch.set_num_threads(1)
    def load_native(run):
        """CERL or RMR checkpoint with the word classes of its own training prefix."""
        rc = json.loads((ROOT/f'experiments/results/token_language/{run}.json').read_text()); g = rc['args']
        tr = C.load_fineweb(ROOT/'data/fineweb_gpt2/fineweb_train_000001.bin'); n = min(g['train_tokens']//g['lane']*g['lane'], 4194304*4)
        cls, words = C.word_classes(np.bincount(np.asarray(tr[:g['train_tokens']//g['lane']*g['lane']]), minlength=50257).astype(float), g['classes'])
        if 'residual-memory' in rc.get('stage', ''):                                              # RMR (theory 172): CERL readout over a residual memory; winners recorded by the base race
            import r1_rmr as RM
            cerl = dict(d=g['d'], modes=g['modes'], layers=g['layers'], dk=g['dk'], units=g['units'], topk=g['topk'], explore=g['explore'], product_keys=bool(g.get('product_keys', 0)),
                        level1=g.get('level1', 'tm'), slots=g.get('slots', 1024), slot_m=g.get('slot_m', 4))
            m = RM.RMR(50257, cls, words, cerl, levels=g['levels'], dk=g['read_dk'], top_m=g['top_m'], chunk=g['chunk'], read_topk=g.get('read_topk', 0), rezero=bool(g.get('rezero', 1)), old_decay=g.get('old_decay', 0.0), old_per_target=bool(g.get('old_per_target', 0)), write_race=bool(g.get('write_race', 0)), fire=bool(g.get('fire', 0)), old_norm=bool(g.get('old_norm', 0)))
            m.load_state_dict(torch.load(ROOT/rc['checkpoint'])); m.eval(); m.base.record_winner = True
            m.__class__.last_winner = property(lambda self: self.base.last_winner)
        else:
            m = C.CERL(50257, cls, words, g['d'], g['modes'], g['layers'], g['dk'], g['units'], g['topk'], g['explore'], age_slope=g.get('age_slope', 1.5), product_keys=bool(g.get('product_keys', 0)), level1=g.get('level1', 'tm'), slots=g.get('slots', 1024), slot_m=g.get('slot_m', 4))
            m.load_state_dict(torch.load(ROOT/rc['checkpoint'])); m.eval(); m.record_winner = True
        return m
    cm = load_native(a.cerl)
    rd = json.loads((ROOT/f'experiments/results/token_language/{a.dense}.json').read_text()); gd = rd['args']
    if 'native' in rd.get('stage', '') or 'residual-memory' in rd.get('stage', ''):                # native reference: compare two native members
        dnat = load_native(a.dense); dense_rows_fn = lambda w: cerl_rows(dnat, w)[0]
    else:
        dm = DeepKeyedLM(gd['d'], gd['modes'], gd['layers'], gd['heads'], gd['dk'], gd['dropout']); dm.load_state_dict(torch.load(ROOT/rd['checkpoint'])); dm.eval(); dense_rows_fn = lambda w: dense_rows(dm, w)
    kind = vocab_roles(a.vocab); val = C.load_fineweb(ROOT/'data/fineweb_gpt2/fineweb_val_000000.bin'); out = dict(battle='R1', tag=a.tag, cerl=a.cerl, dense=a.dense)
    slices = dict(flagship=[np.asarray(val[p:p + 1025]) for p in range(25165824, 25165824 + 8192, 1024)], score_65528=[np.asarray(val[p:p + 8192]) for p in range(20971520, 20971520 + 8*8192, 8192)])
    with torch.no_grad():
        for name, lanes in slices.items():
            cn, ck, dn, role, pos, seen = [], [], [], [], [], []
            for w in lanes:
                c_, k_ = cerl_rows(cm, w); cn.append(c_); ck.append(k_); dn.append(dense_rows_fn(w))
                x, y = w[:-1], w[1:]; r = kind[y].copy(); r[(r == 1) & (kind[x] == 3)] = 0; role.append(r); pos.append(np.arange(len(y)))
                first = {}; s_ = np.zeros(len(y), bool)
                for i, tok in enumerate(w):
                    if i >= 1: s_[i - 1] = tok in first
                    first.setdefault(tok, i)
                seen.append(s_)
            cn, ck, dn, role, pos, seen = map(np.concatenate, (cn, ck, dn, role, pos, seen)); res = dict(targets=int(len(cn)), cerl=float(cn.mean()), dense=float(dn.mean()))
            res['by_role'] = {ROLES[i]: dict(share=float((role == i).mean()), cerl=float(cn[role == i].mean()), dense=float(dn[role == i].mean())) for i in range(5) if (role == i).any()}
            res['by_history'] = {f'{lo}-{hi}': dict(cerl=float(cn[(pos >= lo) & (pos < hi)].mean()), dense=float(dn[(pos >= lo) & (pos < hi)].mean())) for lo, hi in ((0, 16), (16, 64), (64, 256), (256, 1024), (1024, 8192)) if ((pos >= lo) & (pos < hi)).any()}
            res['by_copyable'] = {k: dict(share=float(m_.mean()), cerl=float(cn[m_].mean()), dense=float(dn[m_].mean())) for k, m_ in (('seen_earlier_in_window', seen), ('novel_in_window', ~seen))}
            res['by_winning_cause'] = {nm: dict(share=float((ck == i).mean()), cerl=float(cn[ck == i].mean()), dense=float(dn[ck == i].mean())) for i, nm in enumerate(('background', 'concept', 'repeat', 'continue')) if (ck == i).any()}
            out[name] = res; print(name, json.dumps(res), flush=True)
    with (ROOT/f'experiments/results/diagnostics/{a.tag}.json').open('x') as f: json.dump(out, f, indent=1); f.write('\n')


if __name__ == '__main__':
    main()
