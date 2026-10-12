"""B1 native deep parameter learning, trained by causal future-loss interventions."""
import argparse,hashlib,json,resource,sys,time
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch
from torch.func import functional_call
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'experiments/tpp'))
import race_tpp_v19 as native
from temporal_scan_cpu import layer_forward,extension
from types import MethodType
from event_parameter_credit import install_event_hooks
from operator_event_credit import EventOptimizerCredit,packet

class Actor(native.RaceTPP):
    def __init__(self,*args):
        super().__init__(*args);self.cut=False
    def forward(self,t,m,mask):
        tl,ml,count,_=self.loglik(t,m,mask);return -(tl+ml)/count
    def encode(self,t,m,mask):
        dt=torch.diff(t,dim=1,prepend=t[:,:1]).clamp_min(0)/self.scale;age=(t-t[:,:1])/self.scale
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.cut):
            h=self.embed(m)+self.gap(torch.stack([torch.log1p(dt),(dt>0).to(t.dtype),torch.log1p(age)],-1));features=[h.detach().mean((0,1))];self.event_states=[h.detach().reshape(-1,h.shape[-1]).clone()];self.event_marks=m.detach().clone()
            for layer in self.layers:
                h=layer(h,dt);features.append(h.detach().mean((0,1)));self.event_states.append(h.detach().reshape(-1,h.shape[-1]).clone())
        self.features=torch.cat(features)
        if self.cut:h=h.detach().requires_grad_()
        self.terminal=h
        return h,self.marks(h,m,dt)

class Credit(torch.nn.Module):
    def __init__(self,context_size,parameters):
        super().__init__();self.net=torch.nn.Sequential(torch.nn.Linear(context_size,64),torch.nn.Tanh(),torch.nn.Linear(64,64),torch.nn.Tanh(),torch.nn.Linear(64,parameters))
        torch.nn.init.zeros_(self.net[-1].weight);torch.nn.init.zeros_(self.net[-1].bias)
    def forward(self,context):return self.net(context)

def core_name(name):return name.startswith(('embed.','gap.','layers.'))
def initial(seed,depth,K,scale,qs):
    torch.manual_seed(seed);actor=Actor(K,RUN_WIDTH,RUN_WIDTH//2,depth,2,3,2,0.,scale,qs).eval()
    for layer in actor.layers:layer.forward=MethodType(layer_forward,layer)
    return actor
def context(actor,c,loss,t):return packet(actor,c,loss,t)
def issue(actor,field,features,heads):
    named=dict(actor.named_parameters());direction=field(features);pending,velocity=field.propose(named,direction,getattr(actor,'credit_velocity',None));actor.credit_velocity=velocity;return pending
def copy_issued(actor,pending):
    with torch.no_grad():
        for name,value in actor.named_parameters():value.copy_(pending[name])
def one_block(sequence):
    t,m=sequence;t=torch.tensor(t[:32])[None];m=torch.tensor(m[:32])[None];mask=torch.zeros_like(m,dtype=torch.bool);mask[:,16:]=True
    return t,m,mask

def learn(actor,field,blocks,kind,seed,optimizer=None,evaluate=None):
    if field is not None and not hasattr(actor,'event_handles'):install_event_hooks(actor)
    if optimizer is None and field is not None:optimizer=torch.optim.Adam(field.parameters(),lr=.001)
    if kind=='bp_adam' and not hasattr(actor,'actor_optimizer'):actor.actor_optimizer=torch.optim.AdamW(actor.parameters(),lr=.003,weight_decay=0.)
    if kind not in ['bp','bp_adam'] and not hasattr(actor,'head_optimizer'):actor.head_optimizer=torch.optim.AdamW([v for n,v in actor.named_parameters() if not core_name(n)],lr=.003,weight_decay=0.)
    rng=np.random.default_rng(seed);pending=None;reference=None;pending_logprob=None;version=0;pending_version=None;rows=[];evaluations=[];start=time.monotonic();meta_updates=0;deep_updates=0;optimizer_updates=0;window_receipts=0
    if optimizer is not None:optimizer.zero_grad()
    for index,sequence in enumerate(blocks):
        tick=time.monotonic();actor.collect_events=field is not None;t,m,mask=one_block(sequence);named=dict(actor.named_parameters());before=torch.cat([p.detach().flatten() for n,p in named.items() if core_name(n)])
        teacher=kind=='online' and pending is not None and rng.random()<.25
        if teacher:
            assert pending_version==version;actor.cut=False
            active={n:p if core_name(n) else p.detach().requires_grad_() for n,p in pending.items()};loss=functional_call(actor,active,(t,m,mask))
        else:
            actor.cut=kind not in ['bp','bp_adam'];active=named;loss=actor(t,m,mask)
        assert bool(torch.isfinite(loss));head_names=[n for n in named if not core_name(n)]
        if kind in ['bp','bp_adam']:
            gradients=torch.autograd.grad(loss,tuple(named.values()));norm=torch.stack([g.square().sum() for g in gradients]).sum().sqrt().clamp_min(1e-12);factor=(5/norm).clamp_max(1)
            if kind=='bp_adam':
                actor.actor_optimizer.zero_grad()
                for parameter,gradient in zip(named.values(),gradients):parameter.grad=(factor*gradient).detach()
                actor.actor_optimizer.step()
            else:
                with torch.no_grad():
                    for parameter,gradient in zip(named.values(),gradients):parameter.add_(-.001*factor*gradient)
        else:
            selected=[active[n] for n in head_names]+[actor.terminal]
            gradients=torch.autograd.grad(loss,selected,retain_graph=teacher);heads={n:g.detach() for n,g in zip(head_names,gradients[:-1])};features=context(actor,gradients[-1],loss,t)
            if teacher:
                actor.collect_events=False
                with torch.no_grad():reference_loss=functional_call(actor,reference,(t,m,mask))
                assert bool(torch.isfinite(reference_loss));window_length=min(META_COMMIT_BLOCKS,len(blocks)-(index//META_COMMIT_BLOCKS)*META_COMMIT_BLOCKS);((loss+(loss.detach()-reference_loss.detach())*pending_logprob)/(.25*window_length)).backward();meta_updates+=1;window_receipts+=1
            if kind=='online' and ((index+1)%META_COMMIT_BLOCKS==0 or index==len(blocks)-1):
                if window_receipts:
                    torch.nn.utils.clip_grad_norm_(field.parameters(),5.);optimizer.step();optimizer_updates+=1;version+=1
                optimizer.zero_grad();window_receipts=0
            # All received graphs use one held version; commit averaged receipts every16blocks.
            pending=None
            actor.head_optimizer.zero_grad();head_norm=torch.stack([g.square().sum() for g in heads.values()]).sum().clamp_min(1e-24).sqrt();head_scale=(5/head_norm).clamp_max(1)
            for name in head_names:named[name].grad=(head_scale*heads[name]).detach()
            actor.head_optimizer.step()
            reference={n:v.detach().clone() for n,v in named.items()}
            with torch.set_grad_enabled(kind=='online'):pending=issue(actor,field,features,heads)
            pending_logprob=field.last_logprob
            pending_version=version;copy_issued(actor,pending)
        after=torch.cat([p.detach().flatten() for n,p in actor.named_parameters() if core_name(n)]);change=float((after-before).norm());deep_updates+=int(change>0)
        rows.append(dict(block=index,nll=float(loss.detach()),targets=16,teacher=teacher,deep_parameter_change=change,wall_s=time.monotonic()-tick))
        if evaluate is not None and ((index+1)%128==0 or index==len(blocks)-1):
            evaluation=evaluate(actor);evaluation.update(block=index+1,scored_targets=16*(index+1),elapsed_learning_s=time.monotonic()-start);evaluations.append(evaluation)
    # An unobserved final intervention has no target teaching it.
    actor.cut=False;pending=None
    return dict(wall_s=time.monotonic()-start,scored_targets=16*len(blocks),mean_nll=float(np.mean([r['nll'] for r in rows])),meta_updates=meta_updates,optimizer_updates=optimizer_updates,deep_updates=deep_updates,rows=rows,evaluations=evaluations),optimizer

RUN_WIDTH=16
RUN_FIELD='structured'

def new_credit(actor,depth):return EventOptimizerCredit(RUN_WIDTH,{n:p for n,p in actor.named_parameters() if core_name(n)})

META_COMMIT_BLOCKS=16

def heads_only(actor,blocks,evaluate):
    heads=[p for n,p in actor.named_parameters() if not core_name(n)];before={n:p.detach().clone() for n,p in actor.named_parameters() if core_name(n)};start=time.monotonic();rows=[];evaluations=[]
    for index,sequence in enumerate(blocks):
        tick=time.monotonic();actor.cut=True;actor.collect_events=False;loss=actor(*one_block(sequence));gradients=torch.autograd.grad(loss,heads);norm=torch.stack([g.square().sum() for g in gradients]).sum().clamp_min(1e-24).sqrt();factor=(5/norm).clamp_max(1);actor.head_optimizer.zero_grad()
        for p,g in zip(heads,gradients):p.grad=(factor*g).detach()
        actor.head_optimizer.step();assert bool(torch.isfinite(loss));rows.append(dict(block=index,nll=float(loss.detach()),targets=16,teacher=False,deep_parameter_change=0.,wall_s=time.monotonic()-tick))
        if (index+1)%128==0 or index==len(blocks)-1:
            ev=evaluate(actor);ev.update(block=index+1,scored_targets=16*(index+1),elapsed_learning_s=time.monotonic()-start);evaluations.append(ev)
    assert all(torch.equal(p,before[n]) for n,p in actor.named_parameters() if core_name(n));return dict(wall_s=time.monotonic()-start,scored_targets=16*len(blocks),mean_nll=float(np.mean([r['nll'] for r in rows])),meta_updates=0,optimizer_updates=0,deep_updates=0,rows=rows,evaluations=evaluations,core_bitwise_frozen=True)

def main():
    global RUN_WIDTH,META_COMMIT_BLOCKS
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);ap.add_argument('--mode',choices=['smoke','fit'],required=True);ap.add_argument('--seed',type=int,required=True);a=ap.parse_args();RUN_WIDTH=32;out=ROOT/f'experiments/results/credit/{a.tag}.json';assert not out.exists();torch.set_num_threads(1);torch.set_default_dtype(torch.float64);start=time.monotonic();tick=time.monotonic();extension();startup=time.monotonic()-tick
    if a.mode=='fit':
        gate=json.loads((ROOT/'experiments/results/credit/curie_native_operator_event_20261012_v1_smoke.json').read_text());assert gate['gate']
        for name,digest in gate['source_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    parents={};selected=(None,1,'operator_recursive');META_COMMIT_BLOCKS=1
    train=[s for s in native.load_split('taxi','train') if len(s[1])>=32];assert len(train)==1400;validation=[s for s in native.load_split('taxi','dev') if len(s[1])>=32][:128];assert len(validation)==128;gaps=np.concatenate([np.diff(t[:32]) for t,_ in train[:32]]);gaps=gaps[gaps>0];scale=float(np.median(gaps));qs=np.log(np.quantile(gaps,[.1,.5,.9])).tolist();K=1+max(int(m.max()) for _,m in train)
    rng=np.random.default_rng(99920);schedule=[train[int(i)] for _ in range(3) for i in rng.permutation(len(train))];acquisition=4 if a.mode=='smoke' else 128;scored=8 if a.mode=='smoke' else 512;depths=[8] if a.mode=='smoke' else [8,16]
    schedule=train[1328:1340] if a.mode=='smoke' else train[256:896]
    def evaluate(actor):
        tick=time.monotonic();actor.cut=False;actor.collect_events=False
        with torch.no_grad():values=[float(actor(*one_block(sequence))) for sequence in validation]
        assert all(np.isfinite(values));return dict(dev_nll=float(np.mean(values)),per_sequence=values,evaluation_wall_s=time.monotonic()-tick,scored_dev_targets=16*len(values))
    result_rows=[];checkpoints={}
    for depth in depths:
        actor=initial(a.seed,depth,K,scale,qs);original=deepcopy(actor.state_dict());field=new_credit(actor,depth);acquired,meta=learn(actor,field,schedule[:acquisition],'online',a.seed+1);shared_actor=deepcopy(actor.state_dict());shared_field=deepcopy(field.state_dict());shared_meta=deepcopy(meta.state_dict());shared_head=deepcopy(actor.head_optimizer.state_dict());shared_velocity=deepcopy(actor.credit_velocity);arms={}
        for arm in ['online','frozen','heads','bp_adam']:
            actor=initial(a.seed,depth,K,scale,qs)
            if arm=='bp_adam':
                actor.load_state_dict(original);cost,_=learn(actor,None,schedule[:acquisition],arm,a.seed+1);credit=None;optimizer=None
            else:
                actor.load_state_dict(shared_actor);actor.credit_velocity=deepcopy(shared_velocity);credit=new_credit(actor,depth);credit.load_state_dict(shared_field);optimizer=torch.optim.Adam(credit.parameters(),lr=.001);optimizer.load_state_dict(shared_meta);actor.head_optimizer=torch.optim.AdamW([v for n,v in actor.named_parameters() if not core_name(n)],lr=.003,weight_decay=0.);actor.head_optimizer.load_state_dict(shared_head);cost=acquired
            before=evaluate(actor)
            if arm=='heads':measured=heads_only(actor,schedule[acquisition:acquisition+scored],evaluate)
            else:measured,_=learn(actor,credit,schedule[acquisition:acquisition+scored],arm,a.seed+2,optimizer,evaluate)
            measured.update(acquisition=cost,acquisition_dev=before,whole_learning_wall_s=cost['wall_s']+measured['wall_s']+startup,final_dev_nll=measured['evaluations'][-1]['dev_nll'],native_parameters=sum(p.numel() for p in actor.parameters()),credit_parameters=0 if arm in ['bp_adam','heads'] else sum(p.numel() for p in credit.parameters()));arms[arm]=measured;checkpoints[f'{depth}/{arm}']=dict(actor=actor.state_dict(),credit=None if arm in ['bp_adam','heads'] else credit.state_dict());print(json.dumps(dict(depth=depth,arm=arm,final_dev_nll=measured['final_dev_nll'],whole_learning_wall_s=measured['whole_learning_wall_s'],prequential_nll=measured['mean_nll'])),flush=True)
        target=arms['bp_adam']['final_dev_nll'];reached=[v for v in arms['online']['evaluations'] if v['dev_nll']<=target];crossing=arms['online']['acquisition']['wall_s']+startup if arms['online']['acquisition_dev']['dev_nll']<=target else None if not reached else arms['online']['acquisition']['wall_s']+startup+reached[0]['elapsed_learning_s'];ratio=None if crossing is None else arms['bp_adam']['whole_learning_wall_s']/crossing
        result_rows.append(dict(depth=depth,arms=arms,strong_reference_final_target=target,online_first_checkpoint_work_to_target_s=crossing,matched_quality_wall_ratio=ratio,online_gain_vs_frozen=arms['frozen']['final_dev_nll']-arms['online']['final_dev_nll'],online_gain_vs_matched_heads=arms['heads']['final_dev_nll']-arms['online']['final_dev_nll'],online_gain_vs_adam=target-arms['online']['final_dev_nll']))
    gate=all(r['arms']['online']['deep_updates']>0 and r['arms']['online']['meta_updates']>0 for r in result_rows);assert gate;cp=out.with_suffix('.pt');torch.save(checkpoints,cp);sources=['experiments/credit/run_native_operator_event_credit.py','experiments/credit/NATIVE_OPERATOR_EVENT_CREDIT_PROTOCOL.md','experiments/credit/operator_event_credit.py','experiments/credit/batched_event_parameter_credit.py','experiments/credit/event_optimizer_credit.py','experiments/credit/event_parameter_credit.py','experiments/tpp/race_tpp_v19.py','experiments/tpp/temporal_scan_cpu.py','experiments/tpp/temporal_scan_cpu.cpp'];record=dict(status='completed',battle='B1',tag=a.tag,gate=gate,mode=a.mode,selected_recipe=selected[2],parent_sha256=parents,rows=result_rows,config=dict(seed=a.seed,width=32,modes=16,depths=depths,train_sequences=1400,dev_sequences=128,targets_per_block=16,history_per_block=16,acquisition_blocks=acquisition,reuse_blocks=scored,train_passes=1 if a.mode=='fit' else None,meta_commit_blocks=META_COMMIT_BLOCKS,teacher_probability=.25,head_AdamW_lr=.003,credit_Adam_lr=.001,core_step_max=.003,core_norm_cap=.2),wall_s=time.monotonic()-start,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},data_sha256={split:hashlib.sha256((ROOT/f'data/easytpp/taxi/{split}.json').read_bytes()).hexdigest() for split in ['train','dev']},checkpoint=str(cp.relative_to(ROOT)),checkpoint_sha256=hashlib.sha256(cp.read_bytes()).hexdigest(),scope='First-pass operator-conditioned recursive depth credit, actual current-event producer adjoints with incoming temporal state held fixed, plus future-trained temporal/event residuals. Fresh seed/depth8/16 and128acquisition512reuse. 128DEV sequences,2048scoredtargets; no TEST. Allarms10240fittingtargets; same acquired core/head/velocity for online/frozen/heads, heads bitwisecorefreeze. NativeAdamW separate exact training on identical schedule. No further tuning. Firstcheckpoint cost to prespecified-comparator finalquality is an upperbound, unreachedNone. CompleteCPU wall includes acquisition, teacher/reference/metaoptimizer/packet/query/scoring; no FLOP/energy claim. Dense internal numerical writes, independent sequence resets, no optional forward-route credit. Not identical to AWS all31target scoring, so no direct NLL comparison.');out.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(dict(status='completed',wall_s=record['wall_s'],gate=gate)),flush=True)
if __name__=='__main__':main()
