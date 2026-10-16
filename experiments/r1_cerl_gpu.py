#!/usr/bin/env python3
"""R1 GPU scaling driver for the native causal event race member (r1_cerl.CERL unchanged; theory/169).

The GPU ladder fields this member (founder direction 16 Oct: build our own models; experiments/runbooks/GPU_SCALING_LADDER.md).
One pass over fresh lanes of the FineWeb GPT-2 shards (fineweb_train_*.bin in name order from offset 0, seeded permutation),
bf16 autocast on CUDA (word laws and race normalisers in fp32 through log_softmax/logsumexp), warmup + cosine schedule,
exact resume and SIGTERM checkpoints. Evaluation: selection lanes (8 x 2048 @10,485,760) at log-spaced steps with native share
by cause; the 65,528-target slice (8 x 8192 @20,971,520) at 1/4, 1/2 and the end; flagship windows (8 x 1024 @25,165,824).
Measured training FLOPs per target, active work per target, and the capacity axis (concept units, product-key wake).
"""
import argparse, hashlib, json, math, os, signal, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
ROOT = Path(__file__).resolve().parents[1]; sys.path[:0] = [str(ROOT/'experiments'), str(ROOT/'experiments/tpp')]
from r1_cerl import CERL, word_classes, lanes_nll, work_per_target  # noqa: E402
from r1_gpu_ladder import Lanes  # noqa: E402

EXECUTION_ONLY = ('max_steps', 'checkpoint_every', 'log_every', 'row_chunk', 'out_dir', 'data_dir')


def main():
    ap = argparse.ArgumentParser(); A = ap.add_argument
    A('--tag', required=True); A('--seed', type=int, default=0); A('--d', type=int, default=256); A('--modes', type=int, default=128); A('--layers', type=int, default=2)
    A('--dk', type=int, default=64); A('--units', type=int, default=65536); A('--topk', type=int, default=4); A('--explore', type=int, default=2); A('--product-keys', type=int, default=1)
    A('--cause-messages', type=int, default=0); A('--classes', type=int, default=256); A('--lane', type=int, default=1024); A('--batch', type=int, default=32)
    A('--train-tokens', type=int, default=0); A('--budget-flops', type=float, default=0.0); A('--lr', type=float, default=2e-3); A('--wd', type=float, default=0.01); A('--warmup', type=float, default=0.02)
    A('--evals', type=int, default=16); A('--log-every', type=int, default=50); A('--checkpoint-every', type=int, default=500); A('--max-steps', type=int, default=0); A('--row-chunk', type=int, default=512)
    A('--data-dir', default=str(ROOT/'data/fineweb_gpt2')); A('--out-dir', default=str(ROOT/'experiments/results/gpu_ladder'))
    a = ap.parse_args(); torch.manual_seed(a.seed); dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if dev.type == 'cpu': torch.set_num_threads(1)
    else: torch.backends.cuda.matmul.allow_tf32 = True
    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True); out = out_dir/f'{a.tag}.json'; state_path = out_dir/f'{a.tag}.resume.pt'; assert not out.exists()
    val = np.memmap(Path(a.data_dir)/'fineweb_val_000000.bin', dtype=np.uint16, mode='r', offset=1024)
    # word classes from the first 100M training tokens (training data only; fixed for the run)
    first = np.memmap(Path(a.data_dir)/'fineweb_train_000001.bin', dtype=np.uint16, mode='r', offset=1024)
    counts = np.zeros(50257); step_ = 10_000_000
    for i in range(0, min(len(first), 100_000_000), step_): counts += np.bincount(first[i:i + step_], minlength=50257)
    cls, words = word_classes(counts, a.classes)
    model = CERL(50257, cls, words, a.d, a.modes, a.layers, a.dk, a.units, a.topk, a.explore, product_keys=bool(a.product_keys), cause_messages=bool(a.cause_messages)).to(dev)
    params = sum(p.numel() for p in model.parameters())
    from torch.utils.flop_counter import FlopCounterMode
    x0 = torch.randint(0, 50257, (1, a.lane + 1), device=dev)
    with FlopCounterMode(display=False) as fc, torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == 'cuda'):
        model.lane_nll(x0[:, :-1], x0[:, 1:]).mean().backward()
    model.zero_grad(set_to_none=True); fpt = fc.get_total_flops()/a.lane
    train_tokens = a.train_tokens or int(a.budget_flops/fpt); tps = a.batch*a.lane; steps = max(1, train_tokens//tps); n_lanes = steps*a.batch
    data = Lanes(a.data_dir, a.lane + 1, n_lanes); order = np.random.default_rng(a.seed).permutation(n_lanes); warm = max(1, int(a.warmup*steps))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd, fused=dev.type == 'cuda')
    lr_at = lambda s: a.lr*min(1.0, (s + 1)/warm)*(0.1 + 0.9*0.5*(1 + math.cos(math.pi*min(1.0, max(0, s - warm)/max(1, steps - warm)))))
    evals = sorted(set(int(round(steps**(i/(a.evals - 1)))) for i in range(a.evals)) | {steps}); miles = {max(1, steps//4), max(1, steps//2), steps}
    sel_l = np.asarray(val[10485760:10485760 + 8*2048]).reshape(8, 2048); sc_l = np.asarray(val[20971520:20971520 + 8*8192]).reshape(8, 8192)
    fl_l = np.stack([np.asarray(val[p:p + 1025]) for p in range(25165824, 25165824 + 8192, 1024)])
    ident = {k: v for k, v in vars(a).items() if k not in EXECUTION_ONLY}; st = dict(step=0, losses=[], evals=[], milestones=[], wall=0.0, resumes=0)
    if state_path.exists():
        s = torch.load(state_path, map_location=dev, weights_only=False); assert s['ident'] == ident
        model.load_state_dict(s['model']); opt.load_state_dict(s['opt']); torch.set_rng_state(s['rng']); st = s['st']; st['resumes'] += 1
        if dev.type == 'cuda' and s.get('cuda_rng') is not None: torch.cuda.set_rng_state(s['cuda_rng'])
    stop = {'f': False}; signal.signal(signal.SIGTERM, lambda *_: stop.update(f=True))

    def save():
        tmp = state_path.with_suffix('.tmp'); torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state() if dev.type == 'cuda' else None,
                                                            st=st, ident=ident), tmp); os.replace(tmp, state_path)

    def ev(block):                                                                                  # lanes_nll runs on CPU tensors: score on device through a thin wrapper
        model.eval(); tot = n = 0.0; acc = {}
        with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == 'cuda'):
            for i in range(len(block)):
                w = torch.from_numpy(block[i:i + 1].astype(np.int64)).to(dev); x, y = w[:, :-1], w[:, 1:]; L = x.shape[1]
                msg = model.messages(x, y) if model.cause_msgs else None
                for r0 in range(0, L, a.row_chunk):
                    nll, stt = model.lane_nll(x, y, (r0, min(r0 + a.row_chunk, L)), stats=True, msg=msg); tot += nll.float().sum().item(); n += nll.numel()
                    for k, v in stt.items(): acc[k] = acc.get(k, 0.0) + v*nll.numel()
        model.train(); r = dict(nll=tot/n, targets=int(n), native={k: v/n for k, v in acc.items()}); r['native']['native_gain'] = r['native']['bg_only_nll'] - r['nll']; return r
    began = time.time() - st['wall']; t_log = time.time()
    while st['step'] < steps and not stop['f']:
        s = st['step']
        for g in opt.param_groups: g['lr'] = lr_at(s)
        ids = order[s*a.batch:(s + 1)*a.batch]; w = torch.from_numpy(data.get(np.sort(ids))).to(dev, non_blocking=True); x, y = w[:, :-1], w[:, 1:]
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == 'cuda'):
            nll = model.lane_nll(x, y); loss = nll.float().mean(); obj = loss + (model.aux.float().mean() if model.aux is not None else 0.0)
        opt.zero_grad(set_to_none=True); obj.backward(); gn = nn.utils.clip_grad_norm_(model.parameters(), 1.0).item(); opt.step(); st['step'] += 1; s1 = st['step']; st['losses'].append(loss.item())
        if s1 % a.log_every == 0:
            dt = time.time() - t_log; t_log = time.time()
            print(json.dumps(dict(step=s1, of=steps, loss=float(np.mean(st['losses'][-a.log_every:])), grad_norm=gn, tokens_per_s=a.log_every*tps/dt, elapsed_s=round(time.time() - began),
                                  gpu_mem_gb=torch.cuda.max_memory_allocated()/2**30 if dev.type == 'cuda' else None)), flush=True)
        if s1 in evals: st['evals'].append(dict(step=s1, tokens=s1*tps, flops=fpt*s1*tps, select=ev(sel_l), train_recent=float(np.mean(st['losses'][-50:])))); print(json.dumps(st['evals'][-1]), flush=True)
        if s1 in miles: st['milestones'].append(dict(step=s1, tokens=s1*tps, score_65528=ev(sc_l)))
        if s1 % a.checkpoint_every == 0 or s1 == steps: st['wall'] = time.time() - began; save()
        if a.max_steps and s1 >= a.max_steps: break
    st['wall'] = time.time() - began; save()
    if st['step'] < steps:
        print(json.dumps(dict(paused_at_step=st['step'])), flush=True); sys.exit(75 if stop['f'] else 0)
    torch.save(model.state_dict(), out_dir/f'{a.tag}.pt'); n8 = max(1, steps//8)
    res = dict(status='completed', battle='R1', stage='native causal event race member (theory 169), GPU one pass', tag=a.tag, args=vars(a), parameters=params, units=a.units, train_flops_per_target=fpt,
               train_tokens=steps*tps, train_flops=fpt*steps*tps, steps=steps, train_wall_s=st['wall'], resumes=st['resumes'], evals=st['evals'], milestones=st['milestones'],
               prequential_by_eighth=[float(np.mean(st['losses'][i:i + n8])) for i in range(0, steps, n8)], select=ev(sel_l), score_65528=st['milestones'][-1]['score_65528'], flagship=ev(fl_l),
               work_per_target_at_history_1023=work_per_target(model, 1023), device=str(dev), gpu=torch.cuda.get_device_name(0) if dev.type == 'cuda' else None,
               source_sha256={n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ('experiments/r1_cerl_gpu.py', 'experiments/r1_cerl.py', 'experiments/r1_deep_keyed_lm.py')})
    out.write_text(json.dumps(res, indent=1) + '\n'); state_path.unlink()
    print('RESULT', json.dumps({k: res[k] for k in ('select', 'score_65528', 'flagship', 'train_flops', 'parameters')}), flush=True)


if __name__ == '__main__':
    main()
