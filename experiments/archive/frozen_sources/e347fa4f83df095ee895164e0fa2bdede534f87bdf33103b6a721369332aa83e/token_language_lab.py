"""Small-first CPU laboratory for tokenized persistent temporal models.

Run ONLY via queue/run_safe.sh. Training/dev intervals are explicit, disjoint;
public benchmark validation [0,10485760) is reserved. No test evaluation here.
This is a development fit, not a public Transformer win claim.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import resource
import sys
import time
import numpy as np
import torch
from torch import nn
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sleeping_machines.addressed_event_heads import AddressedEventHeads
from sleeping_machines.fast_native_core import fast_class
from sleeping_machines.token_episodes import token_features, detach
from sleeping_machines.token_readout import TokenReadout


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_tokens(path):
    header = np.fromfile(path, dtype='<i4', count=256)
    if len(header) != 256 or tuple(header[:2]) != (20240520, 1):
        raise ValueError('Expected pinned GPT-2 llm.c shard header')
    count = int(header[2])
    if Path(path).stat().st_size != 1024 + 2 * count:
        raise ValueError('Truncated or oversized token shard')
    return np.memmap(path, dtype='<u2', mode='r', offset=1024, shape=(count,))


class Model(nn.Module):
    def __init__(self, args, order):
        super().__init__()
        self.core = fast_class(AddressedEventHeads)(sources=1, content_dim=50257, classes=1,
              payload=args.payload, depth=args.depth, heads=args.heads, pool=args.pool)
        # Unused old classifier has no role or optimizer work in this model.
        self.core.head = nn.Identity()
        if args.tie_pools:
            for layer in self.core.units:
                for head in layer:
                    for pool in head:
                        for unit in pool[1:]:
                            for name in ('input', 'output', 'gate', 'control'):
                                setattr(unit, name, getattr(pool[0], name))
        cuts = (2000, 10000, 30000) if args.readout == 'adaptive' else ()
        self.readout = TokenReadout(self.core.total_payload, 50257, cuts, order)

    def forward(self, ids, state, generator, args, deterministic=False):
        x, st, pis = token_features(self.core, ids, state, generator,
                 route_credit=args.route_credit, deterministic=deterministic, eos=50256,
                 free_bias=args.free_bias, temperature=1. if deterministic else args.temperature)
        return x, st, pis


def interval_tensor(data, start, count, lanes):
    usable = count // lanes
    if usable < 2 or start < 0 or start + count > len(data):
        raise ValueError('Invalid lane/data interval')
    return torch.from_numpy(np.array(data[start:start + usable * lanes], dtype=np.int64)).reshape(lanes, usable)


@torch.no_grad()
def evaluate(model, data, args):
    model.eval(); st = None; total = 0.; count = 0
    gen = torch.Generator().manual_seed(args.seed + 100000)
    for i in range(0, data.shape[1]-1, args.chunk):
        n = min(args.chunk, data.shape[1]-1-i)
        x, st, _ = model(data[:, i:i+n], st, gen, args, deterministic=args.deterministic_eval)
        total += float(model.readout.nll(x, data[:, i+1:i+n+1]).sum()); count += data.shape[0]*n
    model.train()
    return total / count


def usage(writes):
    p = writes.double() / writes.sum(-1, keepdim=True).clamp_min(1)
    return dict(write_counts=writes.tolist(), used_fraction=float((writes > 0).double().mean()),
                effective_receivers=(-torch.where(p > 0, p * p.clamp_min(1e-300).log(), 0.).sum(-1)).exp().tolist(),
                max_write_share=p.max(-1).values.tolist())


@torch.no_grad()
def memory_rank(memories):
    values = []
    for mem in memories:
        # Lane/receiver state covariance; participation alone is insufficient.
        flat = mem.permute(1, 0, 2, 3).flatten(1, 2).double()
        sv = torch.linalg.svdvals(flat - flat.mean(1, keepdim=True))
        energy = sv.square(); prob = energy / energy.sum(-1,keepdim=True).clamp_min(1e-30)
        rank = (-torch.where(prob > 0, prob*prob.clamp_min(1e-30).log(),0.).sum(-1)).exp()
        values.append(rank.tolist())
    return values


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tag', required=True)
    p.add_argument('--train-file', default='data/fineweb_gpt2/fineweb_train_000001.bin')
    p.add_argument('--dev-file', default='data/fineweb_gpt2/fineweb_val_000000.bin')
    p.add_argument('--train-tokens', type=int, default=8192)
    p.add_argument('--dev-offset', type=int, default=20971520)
    p.add_argument('--dev-tokens', type=int, default=1024)
    p.add_argument('--steps', type=int, default=64)
    p.add_argument('--chunk', type=int, default=16)
    p.add_argument('--lanes', type=int, default=8)
    p.add_argument('--payload', type=int, default=16)
    p.add_argument('--heads', type=int, default=2)
    p.add_argument('--depth', type=int, default=2)
    p.add_argument('--pool', type=int, default=4)
    p.add_argument('--lr', type=float, default=.003)
    p.add_argument('--weight-decay', type=float, default=.01)
    p.add_argument('--route-credit', choices=['linear','none'], default='linear')
    p.add_argument('--readout', choices=['adaptive','dense'], default='adaptive')
    p.add_argument('--free-bias', type=float, default=0.)
    p.add_argument('--temperature', type=float, default=1.)
    p.add_argument('--balance', type=float, default=0.)
    p.add_argument('--tie-pools', action='store_true')
    p.add_argument('--deterministic-eval', action='store_true')
    p.add_argument('--seed', type=int, default=6)
    p.add_argument('--eval-every', type=int, default=16)
    p.add_argument('--resume')
    a = p.parse_args()
    if min(a.steps,a.chunk,a.lanes,a.eval_every) < 1 or a.temperature <= 0:
        p.error('Positive budgets and temperature required')
    if a.dev_offset < 10485760:
        p.error('Public benchmark validation is reserved')
    torch.set_num_threads(1); torch.manual_seed(a.seed)
    out = ROOT/'experiments/results/token_language'/a.tag
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.with_suffix('.json').exists():
        raise FileExistsError('Never overwrite a completed run')
    tr = load_tokens(a.train_file); dv = load_tokens(a.dev_file)
    train = interval_tensor(tr, 0, a.train_tokens, a.lanes)
    dev = interval_tensor(dv, a.dev_offset, a.dev_tokens, a.lanes)
    if int(train.max()) > 50256 or int(dev.max()) > 50256:
        raise ValueError('GPT-2 token range violated')
    # Adaptive rank map uses this run's training interval ONLY.
    counts = torch.bincount(train.flatten(), minlength=50257)
    order = torch.argsort(counts, descending=True, stable=True)
    model = Model(a, order)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    gen = torch.Generator().manual_seed(a.seed + 1)
    state = None; cursor = 0; start_step = 0; elapsed = 0.; curve = []; best = math.inf
    writes = torch.zeros(a.depth,a.heads,a.pool,dtype=torch.long)
    identity = dict(train_sha256=sha(a.train_file), dev_sha256=sha(a.dev_file),
         source_sha256={str(f):sha(ROOT/f) for f in ('experiments/token_language_lab.py',
         'sleeping_machines/token_episodes.py','sleeping_machines/token_readout.py',
         'sleeping_machines/recruit_layer.py','sleeping_machines/compiled_episodes.py',
         'sleeping_machines/fast_native_core.py','sleeping_machines/addressed_event_heads.py',
         'sleeping_machines/parallel_stream_language.py','sleeping_machines/sparse_race_language.py')})
    settings = {k:v for k,v in vars(a).items() if k not in ('tag','resume')}
    if a.resume:
        ck = torch.load(a.resume, weights_only=False)
        if ck['identity'] != identity or ck['settings'] != settings:
            raise ValueError('Exact resume requires identical data, source and settings')
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['optimizer'])
        gen.set_state(ck['generator']); torch.set_rng_state(ck['torch_rng'])
        state,cursor,start_step,elapsed,curve,best,writes = [ck[k] for k in
             ('state','cursor','step','elapsed','curve','best','writes')]
    else:
        curve.append(dict(step=0,train_nll=evaluate(model,train[:,:min(129,train.shape[1])],a),
                          dev_nll=evaluate(model,dev,a)))
    if start_step >= a.steps:
        raise ValueError('Checkpoint already reached the requested budget')
    began = time.monotonic(); train_s = 0.; presentations = 0
    for step in range(start_step+1, a.steps+1):
        if cursor >= train.shape[1]-1:
            cursor = 0; state = None  # true corpus wrap, explicitly reset
        n = min(a.chunk,train.shape[1]-1-cursor)
        before = time.monotonic(); opt.zero_grad(set_to_none=True)
        x,state,pis = model(train[:,cursor:cursor+n],state,gen,a)
        nll = model.readout.nll(x,train[:,cursor+1:cursor+n+1]).mean()
        probs = torch.stack([pi for _,pi in pis])
        by_depth = [torch.stack([pi for depth, pi in pis if depth == d]).mean((0,1)) for d in range(a.depth)]
        penalty = a.pool * torch.stack(by_depth).square().sum(-1).mean()
        loss = nll + a.balance * penalty
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Nonfinite fit; descendants blocked')
        loss.backward()
        grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.))
        if not math.isfinite(grad):
            raise FloatingPointError('Nonfinite gradient')
        receiver_grad = {}
        if step % a.eval_every == 0 or step == a.steps:
            for name, param in model.core.named_parameters():
                if name.startswith('units.') and param.grad is not None:
                    receiver_grad[name] = float(param.grad.norm())
        writes += state['last_writes']; state = detach(state)
        opt.step(); cursor += n; presentations += n*a.lanes; train_s += time.monotonic()-before
        if step % a.eval_every == 0 or step == a.steps:
            row = dict(step=step,batch_nll=float(nll.detach()),gradient_norm=grad,
                train_nll=evaluate(model,train[:,:min(129,train.shape[1])],a),dev_nll=evaluate(model,dev,a),
                route_entropy=float(-(probs.detach()*probs.detach().clamp_min(1e-30).log()).sum(-1).mean()),
                routing=usage(writes),receiver_gradient_norms=receiver_grad,
                memory_effective_rank=memory_rank(state['mem']),wall_s=elapsed+time.monotonic()-began)
            curve.append(row); print(json.dumps(row),flush=True)
            if row['dev_nll'] < best:
                best=row['dev_nll']; torch.save(model.state_dict(),out.with_suffix('.best.pt'))
            ck = dict(model=model.state_dict(),optimizer=opt.state_dict(),state=state,cursor=cursor,step=step,
                 generator=gen.get_state(),torch_rng=torch.get_rng_state(),elapsed=row['wall_s'],curve=curve,
                 best=best,writes=writes,identity=identity,settings=settings)
            tmp=out.with_suffix('.pt.tmp'); torch.save(ck,tmp); tmp.replace(out.with_suffix('.pt'))
    result = dict(status='completed',evidence='development single seed',args=vars(a),identity=identity,
         parameters=sum(p.numel() for p in model.parameters()),best_dev_nll=best,curve=curve,
         presentations_this_invocation=presentations,train_wall_s_this_invocation=train_s,
         train_tokens_per_second=presentations/train_s,wall_s=elapsed+time.monotonic()-began,
         max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,host=platform.node(),
         resource_scope='Measured CPU wall/RSS includes core, all-candidate proposals, readout, backward and AdamW; FLOPs not yet traced',
         protocol='Contiguous lane streams; numeric state carried; credit truncated per chunk; EOS reset; separate development interval; public validation untouched')
    out.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__ == '__main__':
    main()
