"""R1 residual race stack: all-depth readout driver with residual stage composition (--residual).

Versioned copy of multidepth_readout_token_language.py (whose hash completed results bind).
With --residual each race stage adds g_d * stage_d(x) to the stream (sleeping_machines/residual_race_episodes.py);
without it the composition is the original serial chain.

Only the later TRAIN block changes; first-block prior, optimizer and boundaries match.
Small-first CPU laboratory for tokenized persistent temporal models.

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
from sleeping_machines.sparse_counterfactual_episodes import detach
from sleeping_machines.residual_ordered_episodes import token_features, ordered_content
from sleeping_machines.packed_token_core import PackedTokenCore
from sleeping_machines.token_readout import TokenReadout
from sleeping_machines.frequency_token_readout import FrequencyTokenReadout
from sleeping_machines.capacity_token_readout import frequency_readout
from sleeping_machines.paired_route_credit import paired_route_credit
from sleeping_machines.sampled_suffix_utility import sample_positions, suffix_difference
from sleeping_machines.event_credit_sites import sample_site, utility_weight


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


class LaneTokens:
    """Contiguous lanes over a uint16 shard; materialize only the requested chunk."""
    def __init__(self,data,start,count,lanes):
        span=count//lanes
        if span<2 or start<0 or start+count>len(data):
            raise ValueError('Invalid streaming data interval')
        self.data,self.start,self.shape=data,start,(lanes,span)
    def __getitem__(self,index):
        lane,part=index
        if lane != slice(None) or not isinstance(part,slice):
            raise ValueError('Streaming access requires all lanes and a time slice')
        begin,end,stride=part.indices(self.shape[1])
        if stride != 1:raise ValueError('Contiguous token windows required')
        values=[self.data[self.start+i*self.shape[1]+begin:self.start+i*self.shape[1]+end]
                for i in range(self.shape[0])]
        return torch.from_numpy(np.array(values,dtype=np.int64))
    def max(self):
        return int(self.data[self.start:self.start+self.shape[0]*self.shape[1]].max())
    def counts(self):
        counts=np.zeros(50257,dtype=np.int64)
        total=self.shape[0]*self.shape[1]
        for offset in range(0,total,262144):
            chunk=self.data[self.start+offset:self.start+min(offset+262144,total)]
            counts+=np.bincount(chunk,minlength=50257)
        return torch.from_numpy(counts)


class BlockLaneTokens(LaneTokens):
    """Common first block/prior; repeated or fresh later blocks, same boundaries."""
    def __init__(self,data,count,lanes,exposure,steps,chunk):
        super().__init__(data,0,count,lanes)
        self.block=0;self.block_tokens=count;self.exposure=exposure
        self.blocks=math.ceil(steps/math.ceil((self.shape[1]-1)/chunk))
        self.admitted=count*(self.blocks if exposure=='fresh' else 1)
        if self.admitted>len(data):raise ValueError('Insufficient distinct TRAIN tokens')
    def __getitem__(self,index):
        original=self.start
        self.start=self.block*self.block_tokens if self.exposure=='fresh' else 0
        try:return super().__getitem__(index)
        finally:self.start=original
    def max(self):return int(self.data[:self.admitted].max())
    def advance(self):
        if self.block+1>=self.blocks:raise ValueError('Exhausted registered training blocks')
        self.block+=1


def clone_numeric_state(state):
    if state is None:return None
    return {k:([x.detach().clone() for x in v] if isinstance(v,list) else v.detach().clone()) for k,v in state.items()}


def learner_fingerprint(value):
    """Bind actual acquired state, independent of the intervention's name."""
    h=hashlib.sha256()
    def visit(x):
        h.update(type(x).__name__.encode())
        if torch.is_tensor(x):
            h.update(str(x.dtype).encode());h.update(str(tuple(x.shape)).encode())
            h.update(x.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(x,dict):
            for key in sorted(x,key=str):visit(key);visit(x[key])
        elif isinstance(x,(tuple,list)):
            for v in x:visit(v)
        else:h.update(repr(x).encode())
    visit(value);return h.hexdigest()


class Model(nn.Module):
    def __init__(self, args, order, counts):
        super().__init__()
        self.force_site = self.suppress_site = None
        self.shadow_mode=False
        self.replay_window=None
        self.position_generator=torch.Generator().manual_seed(args.seed+4)
        self.site_generator=torch.Generator().manual_seed(args.seed+5)
        self.local_generator=torch.Generator().manual_seed(args.seed+3)
        self.alternative_generator = torch.Generator().manual_seed(args.seed + 2)
        self.core = fast_class(AddressedEventHeads)(sources=1, content_dim=50257, classes=1,
              payload=args.payload, depth=args.depth, heads=args.heads, pool=args.pool)
        # Unused old classifier has no role or optimizer work in this model.
        self.core.head = nn.Identity()
        if args.input_init == 'balanced':
            nn.init.normal_(self.core.content.weight, std=.1)
            nn.init.zeros_(self.core.embedding.weight)
            nn.init.zeros_(self.core.source_gate.weight)
            nn.init.constant_(self.core.source_gate.bias, -2.)
        if args.tie_pools:
            for layer in self.core.units:
                for head in layer:
                    for pool in head:
                        for unit in pool[1:]:
                            for name in ('input', 'output', 'gate', 'control'):
                                setattr(unit, name, getattr(pool[0], name))
        self.core=PackedTokenCore(self.core)
        if getattr(args, 'residual', False):
            init = args.residual_init if args.residual_init is not None else args.depth ** -.5
            self.core.residual_gain = nn.Parameter(torch.full((args.depth, args.payload * args.heads), float(init)))
        cuts = (2000, 10000, 30000) if args.readout == 'adaptive' else ()
        self.readout = (frequency_readout(self.core.total_payload*self.core.depth,counts,cuts,order,args.minimum_tail_width)
            if args.readout_init == 'frequency' else TokenReadout(self.core.total_payload*self.core.depth,50257,cuts,order))

    def forward(self, ids, state, generator, args, deterministic=False):
        # Same numeric stream and optimizer batch; credit boundaries are separate.
        features=[]; probabilities=[]; winners=[]; writes=None
        content=ordered_content(self.core.content.weight,ids)
        for begin in range(0,ids.shape[1],args.credit_window):
            end=min(begin+args.credit_window,ids.shape[1])
            force=mask=None
            if self.force_site is not None and begin<=self.force_site[0]<end:
                force=(self.force_site[0]-begin,*self.force_site[1:])
            if self.suppress_site is not None and begin<=self.suppress_site[0]<end:
                mask=(self.suppress_site[0]-begin,*self.suppress_site[1:])
            if not self.shadow_mode and self.suppress_site is not None and begin<=self.suppress_site[0]<end:
                used = state if args.state_mode=='carry' or begin>0 else None
                self.replay_window = dict(begin=begin, state=clone_numeric_state(used), rng=generator.get_state().clone())
            x,state,pis=token_features(self.core,ids[:,begin:end],
                state if args.state_mode=='carry' or begin>0 or self.shadow_mode else None,generator,
                route_credit=args.route_credit if self.training and not self.shadow_mode and args.future_site=='first' else 'none',
                deterministic=deterministic,eos=50256,free_bias=args.free_bias,
                temperature=1. if deterministic else args.temperature,force_site=force,
                suppress_site=mask,alternative_generator=self.local_generator,compiled=args.compiled,
                token_content=content[begin:end])
            features.append(x);probabilities.extend(pis)
            winners.append(state['race_winners'])
            writes=state['last_writes'] if writes is None else writes+state['last_writes']
            if end<ids.shape[1]:state=detach(state)
        state['race_winners']=torch.cat(winners,0);state['last_writes']=writes
        return torch.cat(features,1),state,probabilities


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
    process_started=time.monotonic()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exposure',choices=['repeated','fresh'],required=True)
    p.add_argument('--tag', required=True)
    p.add_argument('--train-file', default='data/fineweb_gpt2/fineweb_train_000001.bin')
    p.add_argument('--dev-file', default='data/fineweb_gpt2/fineweb_val_000000.bin')
    p.add_argument('--train-tokens', type=int, default=8192)
    p.add_argument('--dev-offset', type=int, default=20971520)
    p.add_argument('--dev-tokens', type=int, default=1024)
    p.add_argument('--steps', type=int, default=64)
    p.add_argument('--chunk', type=int, default=16)
    p.add_argument('--eval-lanes',type=int,default=8)
    p.add_argument('--lanes', type=int, default=8)
    p.add_argument('--credit-window',type=int,default=16,help='State-gradient and paired-utility boundary inside the optimizer chunk')
    p.add_argument('--minimum-tail-width',type=int,default=0)
    p.add_argument('--payload', type=int, default=16)
    p.add_argument('--heads', type=int, default=2)
    p.add_argument('--depth', type=int, default=2)
    p.add_argument('--pool', type=int, default=4)
    p.add_argument('--lr', type=float, default=.003)
    p.add_argument('--weight-decay', type=float, default=.01)
    p.add_argument('--route-credit', choices=['sampled','none'], default='sampled')
    p.add_argument('--compiled',action='store_true')
    p.add_argument('--residual',action='store_true')
    p.add_argument('--residual-init',type=float)
    p.add_argument('--future-site',choices=['first','uniform'],default='uniform')
    p.add_argument('--future-score-positions',type=int,default=0,help='Uniform suffix scoring positions; zero uses the full-score control')
    p.add_argument('--teacher-probability',type=float,default=1.)
    p.add_argument('--future-every',type=int,default=1)
    p.add_argument('--future-scale',type=float,default=1.)
    p.add_argument('--readout-init',choices=['legacy','frequency'],default='legacy')
    p.add_argument('--input-init', choices=['legacy','balanced'], default='legacy')
    p.add_argument('--state-mode', choices=['carry','reset'], default='carry')
    p.add_argument('--readout', choices=['adaptive','dense'], default='adaptive')
    p.add_argument('--free-bias', type=float, default=0.)
    p.add_argument('--temperature', type=float, default=1.)
    p.add_argument('--balance', type=float, default=0.)
    p.add_argument('--tie-pools', action='store_true')
    p.add_argument('--deterministic-eval', action='store_true')
    p.add_argument('--seed', type=int, default=6)
    p.add_argument('--eval-every', type=int, default=16)
    p.add_argument('--resume')
    p.add_argument('--stop-after-step',type=int,help='Diagnostic exact interruption after this optimizer update')
    a = p.parse_args()
    if not 0<a.teacher_probability<=1 or a.future_every!=1 or a.future_site!='uniform':
        raise ValueError('Bernoulli teacher requires probability in(0,1], uniform sites and cadence1')
    if min(a.steps,a.chunk,a.lanes,a.eval_every) < 1 or a.temperature <= 0:
        p.error('Positive budgets and temperature required')
    if a.dev_offset < 10485760:
        p.error('Public benchmark validation is reserved')
    if a.credit_window < 1:
        p.error('Positive credit window required')
    if a.minimum_tail_width < 0 or a.minimum_tail_width > a.payload*a.heads:
        p.error('Tail minimum must lie within the feature width')
    if a.minimum_tail_width and a.readout_init != 'frequency':
        p.error('Selectable tail capacity currently uses frequency initialization')
    if a.future_score_positions < 0 or a.future_every < 0 or a.future_scale < 0 or (a.future_every > 0 and a.pool < 2):
        p.error('Future teacher needs nonnegative controls and at least two receivers')
    torch.set_num_threads(1); torch.manual_seed(a.seed)
    out = ROOT/'experiments/results/token_language'/a.tag
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.with_suffix('.json').exists() or out.with_suffix('.partial.json').exists():
        raise FileExistsError('Never overwrite a completed run')
    tr = load_tokens(a.train_file); dv = load_tokens(a.dev_file)
    train = BlockLaneTokens(tr,a.train_tokens,a.lanes,a.exposure,a.steps,a.chunk)
    dev = interval_tensor(dv, a.dev_offset, a.dev_tokens, a.eval_lanes)
    if int(train.max()) > 50256 or int(dev.max()) > 50256:
        raise ValueError('GPT-2 token range violated')
    # Adaptive rank map uses this run's training interval ONLY.
    counts = train.counts()
    order = torch.argsort(counts, descending=True, stable=True)
    probe = interval_tensor(tr,0,min(8192,a.train_tokens),a.eval_lanes)
    model = Model(a, order, counts)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    gen = torch.Generator().manual_seed(a.seed + 1)
    teacher_gen=torch.Generator().manual_seed(a.seed+6)
    teacher_calls=teacher_replay_tokens=teacher_scored_targets=0
    state = None; cursor = 0; start_step = 0; elapsed = 0.; curve = []; best = math.inf
    total_presentations=0; total_train_s=0.; best_path=None
    writes = torch.zeros(a.depth,a.heads,a.pool,dtype=torch.long)
    identity = dict(train_sha256=sha(a.train_file), dev_sha256=sha(a.dev_file),
         source_sha256={str(f):sha(ROOT/f) for f in (str(Path(__file__).resolve().relative_to(ROOT)),
         'sleeping_machines/multidepth_counterfactual_episodes.py','sleeping_machines/multidepth_ordered_episodes.py','sleeping_machines/residual_race_episodes.py','sleeping_machines/residual_ordered_episodes.py','sleeping_machines/ordered_token_lookup.py','sleeping_machines/ordered_token_episodes.py',
         'sleeping_machines/sparse_counterfactual_episodes.py','sleeping_machines/sparse_counterfactual_layer.py',
         'sleeping_machines/sparse_training.py','sleeping_machines/packed_token_core.py','sleeping_machines/token_readout.py','sleeping_machines/frequency_token_readout.py',
         'sleeping_machines/paired_route_credit.py','sleeping_machines/sampled_suffix_utility.py','sleeping_machines/event_credit_sites.py','sleeping_machines/capacity_token_readout.py',
         'sleeping_machines/recruit_layer.py','sleeping_machines/compiled_episodes.py',
         'sleeping_machines/fast_native_core.py','sleeping_machines/addressed_event_heads.py',
         'sleeping_machines/parallel_stream_language.py','sleeping_machines/sparse_race_language.py')})
    settings = {k:v for k,v in vars(a).items() if k not in ('tag','resume','stop_after_step')}
    if a.resume:
        ck = torch.load(a.resume, weights_only=False)
        if ck['identity'] != identity or ck['settings'] != settings:
            raise ValueError('Exact resume requires identical data, source and settings')
        teacher_gen.set_state(ck['teacher_generator'])
        teacher_calls,teacher_replay_tokens,teacher_scored_targets=ck['teacher_work']
        train.block=ck['training_block']
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['optimizer'])
        model.site_generator.set_state(ck['site_generator']); model.position_generator.set_state(ck['position_generator']); model.local_generator.set_state(ck['local_generator']); model.alternative_generator.set_state(ck['alternative_generator']); gen.set_state(ck['generator']); torch.set_rng_state(ck['torch_rng'])
        state,cursor,start_step,elapsed,curve,best,writes = [ck[k] for k in
             ('state','cursor','step','elapsed','curve','best','writes')]
        total_presentations,total_train_s,best_path=[ck[k] for k in ('total_presentations','total_train_s','best_path')]
    else:
        curve.append(dict(step=0,train_nll=evaluate(model,probe[:,:min(129,probe.shape[1])],a),
                          dev_nll=evaluate(model,dev,a)))
    if start_step >= a.steps:
        raise ValueError('Checkpoint already reached the requested budget')
    began = time.monotonic(); train_s = 0.; presentations = 0
    end_step=min(a.steps,a.stop_after_step) if a.stop_after_step is not None else a.steps
    if end_step <= start_step:raise ValueError('Interruption must be after the restored step')
    for step in range(start_step+1,end_step+1):
        if cursor >= train.shape[1]-1:
            cursor = 0; state = None  # common block boundary in both exposure arms
            train.advance()
        n = min(a.chunk,train.shape[1]-1-cursor)
        before = time.monotonic(); opt.zero_grad(set_to_none=True)
        buf=train[:,cursor:cursor+n+1]
        initial_state = state
        rng_before = gen.get_state()
        use_future = bool(torch.rand((),generator=teacher_gen)<a.teacher_probability)
        depth,head = (step-1)%a.depth, ((step-1)//a.depth)%a.heads
        token,inverse_site_probability=0,1
        if use_future and a.future_site=='uniform':
            token,depth,head,inverse_site_probability=sample_site(n,a.depth,a.heads,model.site_generator)
        model.suppress_site = (token,depth,head) if use_future else None
        x,state,pis = model(buf[:,:n],state,gen,a)
        token_nll = model.readout.nll(x,buf[:,1:])
        nll = token_nll.mean()
        future_term = 0.; future_audit = None
        if use_future:
            pi = pis[token*a.depth+depth][1][:,head,:].double()
            winner = state['race_winners'][token,depth,:,head]
            eligible = 1 - torch.nn.functional.one_hot(winner,a.pool).double()
            conditional = pi.detach()*eligible
            conditional /= conditional.sum(-1,keepdim=True)
            proposal = .9*conditional + .1*eligible/(a.pool-1)
            alt = torch.multinomial(proposal,1,generator=model.alternative_generator).squeeze(1)
            q = proposal.gather(1,alt[:,None]).squeeze(1)
            window = model.replay_window; window_begin = window['begin']
            model.force_site = (token-window_begin,depth,head,alt)
            model.suppress_site = (token-window_begin,depth,head)
            shadow_gen = torch.Generator(); shadow_gen.set_state(window['rng'])
            model.shadow_mode=True
            credit_end=min(n,(token//a.credit_window+1)*a.credit_window)
            positions=sample_positions(credit_end-token,a.future_score_positions,model.position_generator)+token
            replay_length=int(positions[-1])+1
            with torch.no_grad():
                shadow,_,_ = model(buf[:,window_begin:replay_length],window['state'],shadow_gen,a)
                difference = suffix_difference(model.readout,shadow,buf[:,window_begin+1:],token_nll.reshape(a.lanes,n)[:,window_begin:],positions-window_begin)
            model.shadow_mode=False
            credit_weight=inverse_site_probability*(credit_end-token)/n
            future_term = credit_weight*paired_route_credit(pi, alt, q, difference)/a.teacher_probability
            teacher_calls+=1
            teacher_replay_tokens+=(replay_length-window_begin)*a.lanes
            teacher_scored_targets+=len(positions)*a.lanes
            future_audit = dict(site=[token,depth,head],inverse_site_probability=inverse_site_probability,utility_weight=credit_weight,credit_end=credit_end,winner=winner.tolist(),alternative=alt.tolist(),
                               scoring_positions=positions.tolist(),replay_length=replay_length-window_begin,skipped_prefix=window_begin,proposal_probability=q.tolist(),mean_suffix_loss_delta=float(difference.mean()))
        model.force_site = model.suppress_site = None
        model.replay_window = None
        probs = torch.stack([pi for _,pi in pis])
        by_depth = [torch.stack([pi for depth, pi in pis if depth == d]).mean((0,1)) for d in range(a.depth)]
        penalty = a.pool * torch.stack(by_depth).square().sum(-1).mean()
        loss = nll + a.balance * penalty + a.future_scale * future_term
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Nonfinite fit; descendants blocked')
        loss.backward()
        grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.))
        if not math.isfinite(grad):
            raise FloatingPointError('Nonfinite gradient')
        receiver_grad = {}
        if step % a.eval_every == 0 or step == end_step:
            for name, param in model.core.named_parameters():
                if name.startswith('banks.') and param.grad is not None:
                    field=name.split('.')[1]
                    if field=='query':receiver_grad[name]=float(param.grad.norm())
                    else:
                        g=param.grad
                        receiver_grad[name]=(g.norm(dim=tuple(range(3,g.ndim))) if g.ndim>3 else g.abs()).tolist()
        writes += state['last_writes']; state = detach(state)
        opt.step(); cursor += n; presentations += n*a.lanes; total_presentations += n*a.lanes
        duration=time.monotonic()-before; train_s += duration; total_train_s += duration
        if step % a.eval_every == 0 or step == end_step:
            row = dict(step=step,batch_nll=float(nll.detach()),gradient_norm=grad,
                train_nll=evaluate(model,probe[:,:min(129,probe.shape[1])],a),dev_nll=evaluate(model,dev,a),
                future_write_teacher=future_audit,route_entropy=float(-(probs.detach()*probs.detach().clamp_min(1e-30).log()).sum(-1).mean()),
                routing=usage(writes),receiver_gradient_norms=receiver_grad,
                memory_effective_rank=memory_rank(state['mem']),wall_s=elapsed+time.monotonic()-began)
            if step==a.steps//2:
                row['learner_fingerprint']=learner_fingerprint(dict(model=model.state_dict(),optimizer=opt.state_dict(),state=state,
                    generator=gen.get_state(),site=model.site_generator.get_state(),position=model.position_generator.get_state(),
                    local=model.local_generator.get_state(),alternative=model.alternative_generator.get_state(),torch_rng=torch.get_rng_state(),cursor=cursor))
            curve.append(row); print(json.dumps(row),flush=True)
            if row['dev_nll'] < best:
                best=row['dev_nll']; best_path=str(out.with_suffix(f'.best_step{step:08d}.pt'))
                torch.save(model.state_dict(),best_path)
            ck = dict(teacher_generator=teacher_gen.get_state(),teacher_work=[teacher_calls,teacher_replay_tokens,teacher_scored_targets],training_block=train.block,model=model.state_dict(),optimizer=opt.state_dict(),state=state,cursor=cursor,step=step,
                 site_generator=model.site_generator.get_state(),position_generator=model.position_generator.get_state(),local_generator=model.local_generator.get_state(),alternative_generator=model.alternative_generator.get_state(),generator=gen.get_state(),torch_rng=torch.get_rng_state(),elapsed=row['wall_s'],curve=curve,
                 best=best,writes=writes,identity=identity,settings=settings,
                 total_presentations=total_presentations,total_train_s=total_train_s,best_path=best_path)
            tmp=out.with_suffix('.pt.tmp'); torch.save(ck,tmp); tmp.replace(out.with_suffix('.pt'))
    result = dict(teacher_work=dict(admitted_calls=teacher_calls,offered_calls=end_step,replayed_tokens=teacher_replay_tokens,scored_targets=teacher_scored_targets,probability=a.teacher_probability),status='completed' if end_step==a.steps else 'checkpointed',evidence='development single seed',args=vars(a),identity=identity,
         parameters=sum(p.numel() for p in model.parameters()),core_bank_sharing=model.core.tied,
         source_sha256=identity['source_sha256'],process_wall_s=time.monotonic()-process_started,
         exposure_protocol=dict(policy=a.exposure,prior_tokens=a.train_tokens,unique_admitted_tokens=train.admitted,blocks=train.blocks,completed_block=train.block,shared_boundary_reset=True),
         core_parameters=sum(p.numel() for p in model.core.parameters()),
         readout_parameters=sum(p.numel() for p in model.readout.parameters()),
         decoder_tail_widths=getattr(model.readout,'tail_widths',None),best_dev_nll=best,curve=curve,
         presentations_total=total_presentations,train_wall_s_total=total_train_s,best_checkpoint=best_path,
         presentations_this_invocation=presentations,train_wall_s_this_invocation=train_s,
         train_tokens_per_second=presentations/train_s,wall_s=elapsed+time.monotonic()-began,
         max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,host=platform.node(),
         resource_scope='Measured CPU wall/RSS includes core, all-key scoring, winner/sampled-value proposals, winner-only suffix replay, readout, backward and AdamW; FLOPs not yet traced',
         protocol='Packed parameter banks; winner+sampled-value factual learning and winner-only suffix replay; uniform event-site credit (local value surrogate disabled) or first-site control; uniformly sampled causal suffix utility with winner-only replay through the last scored position; bounded per-chunk alternative-write suffix teacher with fixed site first-time/common future noise; contiguous lane streams; numeric state carried; credit truncated per configured window inside the optimizer chunk; EOS reset; separate development interval; public validation untouched')
    out.with_suffix('.json' if end_step==a.steps else '.partial.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__ == '__main__':
    main()
