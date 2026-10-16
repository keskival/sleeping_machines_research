#!/usr/bin/env python3
"""R1 depth per fresh token: the plan-B deep keyed member trained in ONE pass over fresh lanes.

The 1M plan-B ladder (r1_deep_keyed_lm.py, AWS) repeats 1M tokens for up to 20 epochs; the keyed L2 member is already
data-limited there (best epoch 2, train 5.54 vs selection 6.31), so deeper models are judged partly on overfitting. Here
every update sees new lanes, so the training loss before each update is a held-out (prequential) loss and the depth
comparison measures prediction gained per token, the quantity depth is supposed to improve.

Model: DeepKeyedLM from r1_deep_keyed_lm.py unchanged (pre-norm residual blocks with depth-scaled init, exact parallel
temporal memory, per-layer keyed race read, output keyed token-slot bonus). Data: fineweb_train_000001 lanes of 256 tokens
from offset 0 (the same tokens as the keyed 1M/4M runs), visited once in a seeded permutation; AdamW with a one-cycle
schedule over the single pass. --compile-blocks 1 trains through torch.compile'd blocks (same parameters, loss agreement
9.5e-7 and 1.57x per update at L8: curie_plan_b_compile_benchmark_20261015_v1); evaluation always runs the eager blocks. Exact resume: model, optimizer, schedule, RNG and the loss log every --checkpoint-every
updates. Scoring: selection slice at each quarter, then the 65,528-target slice and the flagship 8192-target/1024-window
protocol (both disjoint from training and from the selection slice).
"""
import argparse, hashlib, json, math, os, resource, sys, time, types
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT/'experiments')); sys.path.insert(0, str(ROOT/'experiments/tpp'))
from r1_token_keyed_lm import load, evaluate as evaluate_keyed, lane_nll, V  # noqa: E402
from r1_deep_keyed_lm import DeepKeyedLM  # noqa: E402

FLAGSHIP_START, FLAGSHIP_TARGETS, FLAGSHIP_WINDOW = 25165824, 8192, 1024
EXECUTION_ONLY = ('max_steps', 'compile_blocks')


def compiled_states(self, x):
    """DeepKeyedLM.states through the compiled blocks while training; eager blocks otherwise (scoring shapes vary)."""
    if not self.training: return DeepKeyedLM.states(self, x)
    e = self.embed(x); h = e
    for b in self.train_blocks: h = b(h, e)
    return self.out_norm(h), e


def flagship(model, val):
    model.eval(); total = n = 0
    with torch.no_grad():
        for begin in range(FLAGSHIP_START, FLAGSHIP_START + FLAGSHIP_TARGETS, FLAGSHIP_WINDOW):
            t_, n_ = lane_nll(model, np.asarray(val[begin:begin + FLAGSHIP_WINDOW + 1])[None], 256); total += t_; n += n_
    assert n == FLAGSHIP_TARGETS
    return dict(nll=total/n, targets=n, target_start=FLAGSHIP_START + 1, window=FLAGSHIP_WINDOW)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tag', required=True); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--train-tokens', type=int, default=4194304); ap.add_argument('--lane', type=int, default=256); ap.add_argument('--d', type=int, default=64)
    ap.add_argument('--modes', type=int, default=32); ap.add_argument('--layers', type=int, default=2); ap.add_argument('--heads', type=int, default=4); ap.add_argument('--dk', type=int, default=16)
    ap.add_argument('--dropout', type=float, default=0.1); ap.add_argument('--lr', type=float, default=3e-3); ap.add_argument('--wd', type=float, default=0.01)
    ap.add_argument('--batch', type=int, default=4); ap.add_argument('--checkpoint-blocks', type=int, default=0); ap.add_argument('--checkpoint-every', type=int, default=256)
    ap.add_argument('--compile-blocks', type=int, default=0, help='train through torch.compile\'d blocks (execution only; not part of the resume identity)')
    ap.add_argument('--max-steps', type=int, default=0, help='smoke only: stop after this many updates (no scoring)')
    a = ap.parse_args(); torch.set_num_threads(1); torch.manual_seed(a.seed)
    train_all = load(ROOT/'data/fineweb_gpt2/fineweb_train_000001.bin'); val = load(ROOT/'data/fineweb_gpt2/fineweb_val_000000.bin')
    n_lanes = a.train_tokens//a.lane; train = np.asarray(train_all[:n_lanes*a.lane]).reshape(n_lanes, a.lane)
    order = np.random.default_rng(a.seed).permutation(n_lanes); total = math.ceil(n_lanes/a.batch); quarter = math.ceil(total/4)
    model = DeepKeyedLM(a.d, a.modes, a.layers, a.heads, a.dk, a.dropout); model.checkpoint_blocks = bool(a.checkpoint_blocks); params = sum(p.numel() for p in model.parameters())
    if a.compile_blocks:
        assert not a.checkpoint_blocks, 'compiled blocks run without activation checkpointing'
        model.train_blocks = [torch.compile(b, dynamic=False) for b in model.blocks]   # plain list: shares parameters, state_dict unchanged
        model.states = types.MethodType(compiled_states, model)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd); sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=total, pct_start=0.05)
    out_dir = ROOT/'experiments/results/token_language'; out = out_dir/f'{a.tag}.json'; state_path = out_dir/f'{a.tag}.resume.pt'; assert not out.exists()
    step, losses, milestones, wall, resumes = 0, [], [], 0.0, 0
    if state_path.exists():
        s = torch.load(state_path, weights_only=False); model.load_state_dict(s['model']); opt.load_state_dict(s['opt']); sched.load_state_dict(s['sched'])
        torch.set_rng_state(s['torch_rng']); step, losses, milestones, wall, resumes = s['step'], s['losses'], s['milestones'], s['wall'], s['resumes'] + 1
        assert s['args'] == {k: v for k, v in vars(a).items() if k not in EXECUTION_ONLY}, 'resume with different settings'
        print(json.dumps(dict(resumed_at_step=step)), flush=True)
    began = time.time() - wall
    def save():
        tmp = state_path.with_suffix('.tmp'); torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), torch_rng=torch.get_rng_state(), step=step,
                   losses=losses, milestones=milestones, wall=time.time() - began, resumes=resumes, args={k: v for k, v in vars(a).items() if k not in EXECUTION_ONLY}), tmp); os.replace(tmp, state_path)
    while step < total:
        model.train(); x = torch.from_numpy(train[np.sort(order[step*a.batch:(step + 1)*a.batch])].astype(np.int64)); h, e = model.states(x[:, :-1])
        lg = model.logits(h, e, x[:, :-1], list(range(a.lane - 1))); loss = F.cross_entropy(lg.reshape(-1, V), x[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); step += 1
        losses.append(loss.item())                                    # fresh lanes: loss before this update is held out
        if step % 100 == 0: print(json.dumps(dict(step=step, of=total, prequential_last100=float(np.mean(losses[-100:])), elapsed_s=round(time.time() - began))), flush=True)
        if step % quarter == 0 or step == total:
            milestones.append(dict(step=step, tokens=step*a.batch*a.lane, select_nll=evaluate_keyed(model, val, 10485760, 8, 2048)['nll'], prequential_segment=float(np.mean(losses[-quarter:])), elapsed_s=round(time.time() - began)))
            print(json.dumps(milestones[-1]), flush=True)
        if step % a.checkpoint_every == 0 or step == total: save()
        if a.max_steps and step >= a.max_steps:
            print(json.dumps(dict(smoke_steps=step, s_per_step=(time.time() - began)/step, max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)), flush=True); return
    seg = max(1, total//8)
    result = dict(status='completed', battle='R1', stage='depth per fresh token: plan-B deep keyed member, one pass', tag=a.tag, args=vars(a), parameters=params, steps=step,
                  presentations=step*a.batch*(a.lane - 1), train_wall_s=time.time() - began, resumes=resumes, milestones=milestones,
                  prequential_by_eighth=[float(np.mean(losses[i:i + seg])) for i in range(0, len(losses), seg)], prequential_last_quarter=float(np.mean(losses[-quarter:])),
                  select=evaluate_keyed(model, val, 10485760, 8, 2048), score_65528=evaluate_keyed(model, val, 20971520, 8, 8192), flagship_protocol=flagship(model, val),
                  max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  reference=dict(keyed_L2_4M_multi_epoch_score_65528=5.521061, keyed_L2_4M_flagship=dict(s0=5.624, s1=5.601), p24_flagship=5.978388))
    torch.save(model.state_dict(), out_dir/f'{a.tag}.pt'); result['checkpoint'] = f'experiments/results/token_language/{a.tag}.pt'
    result['source_sha256'] = {n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['experiments/r1_deep_keyed_stream.py', 'experiments/r1_deep_keyed_lm.py', 'experiments/r1_token_keyed_lm.py']}
    out.write_text(json.dumps(result, indent=1) + '\n'); state_path.unlink()
    print('RESULT', json.dumps({k: result[k] for k in ('select', 'score_65528', 'flagship_protocol', 'prequential_last_quarter', 'parameters')}), flush=True)


if __name__ == '__main__':
    main()
