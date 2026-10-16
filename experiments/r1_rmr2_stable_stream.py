#!/usr/bin/env python3
"""Source-bound RMR2 fit with stable long-window decay and complete resume identity.
Reuses the established RMR1 training loop and evaluation protocols; no new recipe.
"""
import argparse,hashlib,json,math,os,resource,sys,time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'experiments'),str(ROOT/'experiments/tpp')]
from r1_cerl import word_classes,load_fineweb,lanes_nll,fineweb_protocols,work_per_target
from r1_rmr2_stable import StableRMR2
EXECUTION_ONLY=('max_steps','checkpoint_every','log_every','row_chunk')
STAGE='RMR2 standing-memory race model, causal stable decay, one pass'
def main():
    ap = argparse.ArgumentParser(); A = ap.add_argument
    A('--mode', choices=('train',), default='train'); A('--tag', default=''); A('--seed', type=int, default=0)
    A('--d', type=int, default=128); A('--modes', type=int, default=64); A('--layers', type=int, default=2); A('--dk', type=int, default=64)
    A('--units', type=int, default=1024); A('--topk', type=int, default=4); A('--explore', type=int, default=2); A('--classes', type=int, default=256)
    A('--levels', type=int, default=2); A('--read-dk', type=int, default=64); A('--top-m', type=int, default=4); A('--chunk', type=int, default=64); A('--read-topk', type=int, default=0); A('--rezero', type=int, default=1)
    A('--train-tokens', type=int, default=4194304); A('--lane', type=int, default=256); A('--batch', type=int, default=4); A('--lr', type=float, default=3e-3); A('--wd', type=float, default=0.01)
    A('--row-chunk', type=int, default=512); A('--checkpoint-every', type=int, default=256); A('--max-steps', type=int, default=0); A('--log-every', type=int, default=100)
    A('--standing',type=int,default=256); A('--standing-m',type=int,default=2); A('--act-cost',type=float,default=0.0); A('--hard-fire-eval',type=int,default=0)
    a = ap.parse_args(); assert a.chunk<=128; torch.set_num_threads(1); torch.manual_seed(a.seed)
    src = {n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ('experiments/r1_rmr2_stable_stream.py','experiments/r1_rmr2_stable.py','experiments/r1_rmr2.py','experiments/r1_rmr.py','experiments/r1_cerl.py','experiments/r1_deep_keyed_lm.py','experiments/r1_token_keyed_lm.py')}
    train_all = load_fineweb(ROOT/'data/fineweb_gpt2/fineweb_train_000001.bin'); val = load_fineweb(ROOT/'data/fineweb_gpt2/fineweb_val_000000.bin')
    n_lanes = a.train_tokens//a.lane; train = np.asarray(train_all[:n_lanes*a.lane]).reshape(n_lanes, a.lane).astype(np.int64)
    cls, words = word_classes(np.bincount(train.reshape(-1), minlength=50257).astype(float), a.classes)
    cerl = dict(d=a.d, modes=a.modes, layers=a.layers, dk=a.dk, units=a.units, topk=a.topk, explore=a.explore)
    model = StableRMR2(50257, cls, words, cerl, levels=a.levels, dk=a.read_dk, top_m=a.top_m, chunk=a.chunk, read_topk=a.read_topk, standing=a.standing,standing_m=a.standing_m,act_cost=a.act_cost,hard_fire_eval=bool(a.hard_fire_eval)); params = sum(p.numel() for p in model.parameters())
    order = np.random.default_rng(a.seed).permutation(n_lanes); total = math.ceil(n_lanes/a.batch); quarter = math.ceil(total/4)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd); sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=total, pct_start=0.05)
    out_dir = ROOT/'experiments/results/token_language'; out = out_dir/f'{a.tag}.json'; state_path = out_dir/f'{a.tag}.resume.pt'; assert a.tag and not out.exists()
    ident = {k: v for k, v in vars(a).items() if k not in EXECUTION_ONLY}; step, losses, milestones, wall, resumes = 0, [], [], 0.0, 0
    if state_path.exists():
        s = torch.load(state_path, weights_only=False); assert s['args'] == ident and s['source_sha256']==src
        model.load_state_dict(s['model']); opt.load_state_dict(s['opt']); sched.load_state_dict(s['sched']); torch.set_rng_state(s['torch_rng'])
        step, losses, milestones, wall, resumes = s['step'], s['losses'], s['milestones'], s['wall'], s['resumes'] + 1
    began = time.time() - wall
    def save():
        tmp = state_path.with_suffix('.tmp'); torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), torch_rng=torch.get_rng_state(), step=step, losses=losses,
                   milestones=milestones, wall=time.time() - began, resumes=resumes, args=ident,source_sha256=src), tmp); os.replace(tmp, state_path)
    sel_l = np.asarray(val[10485760:10485760 + 8*2048]).reshape(8, 2048)
    while step < total:
        model.train(); w = torch.from_numpy(train[np.sort(order[step*a.batch:(step + 1)*a.batch])]); x, y = w[:, :-1], w[:, 1:]
        nll = model.lane_nll(x, y); loss = nll.mean(); obj = loss + (model.aux.mean() if model.aux is not None else 0.0)
        assert bool(torch.isfinite(obj)), 'nonfinite training objective'
        opt.zero_grad(); obj.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); step += 1; losses.append(loss.item())
        if step % a.log_every == 0:
            print(json.dumps(dict(step=step, of=total, prequential_last=float(np.mean(losses[-a.log_every:])), levels=model._mem[2], elapsed_s=round(time.time() - began))), flush=True)
        if step % quarter == 0 or step == total:
            milestones.append(dict(step=step, tokens=step*a.batch*a.lane, select=lanes_nll(model, sel_l, a.row_chunk, stats=True), prequential_segment=float(np.mean(losses[-quarter:])), elapsed_s=round(time.time() - began)))
            print(json.dumps(milestones[-1]), flush=True)
        if step % a.checkpoint_every == 0 or step == total: save()
        if a.max_steps and step >= a.max_steps:
            print(json.dumps(dict(smoke_steps=step, s_per_step=(time.time() - began)/step, max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, parameters=params)), flush=True)
            print(json.dumps(dict(smoke_select_2lanes=lanes_nll(model, sel_l[:2], a.row_chunk, stats=True))), flush=True); return
    from torch.utils.flop_counter import FlopCounterMode
    with FlopCounterMode(display=False) as fc:
        model.train(); w = torch.from_numpy(train[:1]); model.lane_nll(w[:, :-1], w[:, 1:]).mean().backward()
    model.zero_grad(); fpt = fc.get_total_flops()/(a.lane - 1); seg = max(1, total//8)
    result = dict(status='completed', battle='R1', stage=STAGE, tag=a.tag, args=vars(a), parameters=params, steps=step,
                  presentations=step*a.batch*(a.lane - 1), train_wall_s=time.time() - began, resumes=resumes, milestones=milestones, train_flops_per_target=fpt, train_flops=fpt*step*a.batch*(a.lane - 1),
                  prequential_by_eighth=[float(np.mean(losses[i:i + seg])) for i in range(0, len(losses), seg)], prequential_last_quarter=float(np.mean(losses[-quarter:])),
                  **fineweb_protocols(model, val, a.row_chunk), work_level1_per_target=work_per_target(model.base, 255),
                  reference=dict(cerl_4M='curie_r1_cerl_4M_s0_20261016_v1 (flagship 5.4665)', diagnostic_planB_L8_4M_flagship=5.8414), max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, source_sha256=src)
    torch.save(model.state_dict(), out_dir/f'{a.tag}.pt'); result['checkpoint'] = f'experiments/results/token_language/{a.tag}.pt'
    out.write_text(json.dumps(result, indent=1) + '\n'); state_path.unlink()
    print('RESULT', json.dumps({k: result[k] for k in ('select', 'score_65528', 'flagship', 'train_flops_per_target', 'parameters')}), flush=True)


if __name__ == '__main__':
    main()
