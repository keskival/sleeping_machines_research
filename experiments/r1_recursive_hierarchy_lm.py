#!/usr/bin/env python3
"""R1: recursive event depth on the concept hierarchy (theory/RECURSIVE_EVENT_DEPTH.md, P9a-b).

Extends ConceptHierarchyLM (r1_concept_hierarchy_lm.py, unchanged) with concept memories on every level from the clause up
and recursive likelihood heads: at each emission of level k-1 (a completed level-(k-1) whole), level k predicts that
whole's identity mark (top-1 slot of level k-1's concept race; discrete, no gradient: stop-gradient at the mark) and its
length in tokens (categorical over 1..G-1 and an overflow class: a discrete-time hazard model), from the level-k state at
the last token BEFORE that whole began (causal: the prediction never sees the whole it predicts). Heads at sentence,
line and document levels predict clause, sentence and line wholes. The loss is averaged per head and added to the
next-token loss with --recursive-weight. Everything else (levels, future targets, concept memories, tree read option,
one-pass training, resume, scoring) is the concept hierarchy's.
"""
import argparse, hashlib, json, math, os, resource, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT/'experiments')); sys.path.insert(0, str(ROOT/'experiments/tpp'))
import r1_concept_hierarchy_lm as ch  # noqa: E402
import r1_part_whole_lm as pw  # noqa: E402
from r1_token_keyed_lm import load, evaluate as evaluate_keyed, V  # noqa: E402


class RecursiveHead(nn.Module):
    def __init__(self, d, slots, gaps):
        super().__init__(); self.gaps = gaps; self.n = nn.LayerNorm(d); self.mark = nn.Linear(d, slots); self.gap = nn.Linear(d, gaps)


class RecursiveHierarchyLM(ch.ConceptHierarchyLM):
    def __init__(self, d, modes, layers, heads, dk, dropout, horizons=(4, 16, 32, 64, 256), slots=1024, topk=8, concept_levels=(1, 2, 3, 4), tree=False, gaps=64):
        super().__init__(d, modes, layers, heads, dk, dropout, tuple(horizons), slots, topk, tuple(concept_levels), tree)
        # head at level k predicts level k-1 wholes; both levels need concept memories (marks)
        self.recursive = nn.ModuleDict({str(k): RecursiveHead(d, slots, gaps) for k in self.concept_levels if (k - 1) in self.concept_levels})
        self.last_starts = None

    def states(self, x):
        out = super().states(x); self.last_starts = ch.boundaries5(x)[:len(self.levels)]; return out

    def recursive_loss(self):
        """Mean over heads of NLL(mark) + NLL(length) for every completed lower whole that has a predecessor in the lane."""
        total, used = 0., 0
        for key, head in self.recursive.items():
            k = int(key); low = self.last_starts[k - 1]; B, L = low.shape; t = torch.arange(L)
            marks = self.concept_usage[k - 1][..., 0]                                    # top-1 slot at every token (no grad)
            u = self.last_levels[k]; nll = []
            for b in range(B):
                p = t[low[b]]                                                            # starts of level-(k-1) units
                if len(p) < 3: continue
                begin, end = p[1:-1], p[2:]                                              # complete units [begin, end-1] with a predecessor
                y_mark = marks[b, end - 1]; y_gap = (end - begin).clamp_max(head.gaps) - 1
                h = head.n(u[b, begin - 1])                                              # level-k state before the unit began
                nll.append(F.cross_entropy(head.mark(h), y_mark, reduction='none') + F.cross_entropy(head.gap(h), y_gap, reduction='none'))
            if nll: total = total + torch.cat(nll).mean(); used += 1
        return total / max(used, 1), used


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--train-tokens', type=int, default=4194304); ap.add_argument('--lane', type=int, default=1024); ap.add_argument('--d', type=int, default=64)
    ap.add_argument('--modes', type=int, default=32); ap.add_argument('--layers', type=int, default=2); ap.add_argument('--heads', type=int, default=4); ap.add_argument('--dk', type=int, default=16)
    ap.add_argument('--dropout', type=float, default=0.1); ap.add_argument('--lr', type=float, default=3e-3); ap.add_argument('--wd', type=float, default=0.01)
    ap.add_argument('--batch', type=int, default=1); ap.add_argument('--checkpoint-every', type=int, default=256); ap.add_argument('--checkpoint-blocks', type=int, default=0)
    ap.add_argument('--horizons', type=int, nargs='+', default=[4, 16, 32, 64, 256]); ap.add_argument('--future-weight', type=float, default=0.1)
    ap.add_argument('--slots', type=int, default=1024); ap.add_argument('--topk', type=int, default=8); ap.add_argument('--concept-levels', type=int, nargs='*', default=[1, 2, 3, 4])
    ap.add_argument('--tree-read', type=int, default=0); ap.add_argument('--recursive-weight', type=float, default=0.1); ap.add_argument('--gaps', type=int, default=64)
    ap.add_argument('--max-steps', type=int, default=0, help='smoke only: stop after this many updates (no scoring)')
    a = ap.parse_args(); torch.set_num_threads(1); torch.manual_seed(a.seed)
    train_all = load(ROOT/'data/fineweb_gpt2/fineweb_train_000001.bin'); val = load(ROOT/'data/fineweb_gpt2/fineweb_val_000000.bin')
    n_lanes = a.train_tokens//a.lane; train = np.asarray(train_all[:n_lanes*a.lane]).reshape(n_lanes, a.lane)
    order = np.random.default_rng(a.seed).permutation(n_lanes); total = math.ceil(n_lanes/a.batch); quarter = math.ceil(total/4)
    model = RecursiveHierarchyLM(a.d, a.modes, a.layers, a.heads, a.dk, a.dropout, tuple(a.horizons), a.slots, a.topk, tuple(a.concept_levels), bool(a.tree_read), a.gaps)
    model.checkpoint_blocks = bool(a.checkpoint_blocks); params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd); sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=total, pct_start=0.05)
    out_dir = ROOT/'experiments/results/token_language'; out = out_dir/f'{a.tag}.json'; state_path = out_dir/f'{a.tag}.resume.pt'; assert not out.exists()
    step, losses, aux, rec_l, milestones, wall, resumes = 0, [], [], [], [], 0.0, 0
    if state_path.exists():
        s = torch.load(state_path, weights_only=False); model.load_state_dict(s['model']); opt.load_state_dict(s['opt']); sched.load_state_dict(s['sched'])
        torch.set_rng_state(s['torch_rng']); step, losses, aux, rec_l, milestones, wall, resumes = s['step'], s['losses'], s['aux'], s['rec'], s['milestones'], s['wall'], s['resumes'] + 1
        assert s['args'] == {k: v for k, v in vars(a).items() if k != 'max_steps'}, 'resume with different settings'
        print(json.dumps(dict(resumed_at_step=step)), flush=True)
    began = time.time() - wall
    def save():
        tmp = state_path.with_suffix('.tmp'); torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), torch_rng=torch.get_rng_state(), step=step, losses=losses, aux=aux, rec=rec_l,
                   milestones=milestones, wall=time.time() - began, resumes=resumes, args={k: v for k, v in vars(a).items() if k != 'max_steps'}), tmp); os.replace(tmp, state_path)
    while step < total:
        model.train(); x = torch.from_numpy(train[np.sort(order[step*a.batch:(step + 1)*a.batch])].astype(np.int64)); h, e = model.states(x[:, :-1])
        lg = model.logits(h, e, x[:, :-1], list(range(a.lane - 1))); loss = F.cross_entropy(lg.reshape(-1, V), x[:, 1:].reshape(-1))
        fut = model.future_loss(x) if a.future_weight > 0 else torch.zeros(()); rl, _ = model.recursive_loss(); rl = rl if torch.is_tensor(rl) else torch.zeros(())
        opt.zero_grad(); (loss + a.future_weight*fut + a.recursive_weight*rl).backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); step += 1
        losses.append(loss.item()); aux.append(fut.item()); rec_l.append(rl.item())
        if step % 100 == 0: print(json.dumps(dict(step=step, of=total, prequential_last100=float(np.mean(losses[-100:])), future_last100=float(np.mean(aux[-100:])), recursive_last100=float(np.mean(rec_l[-100:])), elapsed_s=round(time.time() - began))), flush=True)
        if step % quarter == 0 or step == total:
            milestones.append(dict(step=step, tokens=step*a.batch*a.lane, select_nll=evaluate_keyed(model, val, 10485760, 8, 2048)['nll'], prequential_segment=float(np.mean(losses[-quarter:])), elapsed_s=round(time.time() - began)))
            print(json.dumps(milestones[-1]), flush=True)
        if step % a.checkpoint_every == 0 or step == total: save()
        if a.max_steps and step >= a.max_steps:
            rec = dict(status='smoke_completed', battle='R1', tag=a.tag, args=vars(a), parameters=params, smoke_steps=step, s_per_step=(time.time() - began)/step,
                       max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, losses=losses, future=aux, recursive=rec_l)
            out.write_text(json.dumps(rec, indent=1) + '\n'); state_path.unlink(missing_ok=True); print(json.dumps({k: rec[k] for k in ('smoke_steps', 's_per_step', 'max_rss_kb')}), flush=True); return
    seg = max(1, total//8)
    result = dict(status='completed', battle='R1', stage='recursive event depth on the concept hierarchy, one pass', tag=a.tag, args=vars(a), parameters=params, steps=step,
                  presentations=step*a.batch*(a.lane - 1), train_wall_s=time.time() - began, resumes=resumes, milestones=milestones,
                  prequential_by_eighth=[float(np.mean(losses[i:i + seg])) for i in range(0, len(losses), seg)], prequential_last_quarter=float(np.mean(losses[-quarter:])),
                  recursive_by_eighth=[float(np.mean(rec_l[i:i + seg])) for i in range(0, len(rec_l), seg)],
                  select=evaluate_keyed(model, val, 10485760, 8, 2048), score_65528=evaluate_keyed(model, val, 20971520, 8, 8192), flagship_protocol=pw.flagship(model, val),
                  max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, reference=dict(no_recursive='curie_r1_concept_4M_l1024_s0_20261015_v1', base='curie_r1_deep_keyed_stream_L2_4M_l1024_s0_20261015_v1'))
    torch.save(model.state_dict(), out_dir/f'{a.tag}.pt'); result['checkpoint'] = f'experiments/results/token_language/{a.tag}.pt'
    result['source_sha256'] = {n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['experiments/r1_recursive_hierarchy_lm.py', 'experiments/r1_concept_hierarchy_lm.py', 'experiments/r1_part_whole_lm.py', 'experiments/r1_deep_keyed_lm.py', 'experiments/tables/gpt2_token_roles.json']}
    out.write_text(json.dumps(result, indent=1) + '\n'); state_path.unlink()
    print('RESULT', json.dumps({k: result[k] for k in ('select', 'score_65528', 'flagship_protocol', 'prequential_last_quarter', 'parameters')}), flush=True)


if __name__ == '__main__':
    main()
