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

class Actor(native.RaceTPP):
    def __init__(self,*args):
        super().__init__(*args);self.cut=False
    def forward(self,t,m,mask):
        tl,ml,count,_=self.loglik(t,m,mask);return -(tl+ml)/count
    def encode(self,t,m,mask):
        dt=torch.diff(t,dim=1,prepend=t[:,:1]).clamp_min(0)/self.scale;age=(t-t[:,:1])/self.scale
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.cut):
            h=self.embed(m)+self.gap(torch.stack([torch.log1p(dt),(dt>0).to(t.dtype),torch.log1p(age)],-1));features=[h.detach().mean((0,1))]
            for layer in self.layers:
                h=layer(h,dt);features.append(h.detach().mean((0,1)))
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
    torch.manual_seed(seed);actor=Actor(K,8,4,depth,2,3,2,0.,scale,qs).eval()
    for layer in actor.layers:layer.forward=MethodType(layer_forward,layer)
    return actor
def context(actor,c,loss,t):
    norm=lambda x:x/x.square().mean().clamp_min(1e-16).sqrt()
    return torch.cat([norm(actor.features),norm(c.detach().mean((0,1))),loss.detach()[None],torch.log1p(torch.diff(t,dim=1).clamp_min(0).mean())[None]]).detach()
def issue(actor,field,features,heads):
    named=dict(actor.named_parameters());names=[n for n in named if core_name(n)];direction=field(features);pieces=direction.split([named[n].numel() for n in names]);g={n:p.reshape_as(named[n]) for n,p in zip(names,pieces)}
    norm=torch.stack([v.square().sum() for v in g.values()]).sum().clamp_min(1e-24).sqrt();factor=(5/norm).clamp_max(1)
    return {n:(value.detach()-.001*factor*g[n]) if core_name(n) else value.detach().clone() for n,value in named.items()}
def copy_issued(actor,pending):
    with torch.no_grad():
        for name,value in actor.named_parameters():value.copy_(pending[name])
def one_block(sequence):
    t,m=sequence;t=torch.tensor(t[:32])[None];m=torch.tensor(m[:32])[None];mask=torch.zeros_like(m,dtype=torch.bool);mask[:,16:]=True
    return t,m,mask

def learn(actor,field,blocks,kind,seed,optimizer=None,evaluate=None):
    if optimizer is None and field is not None:optimizer=torch.optim.Adam(field.parameters(),lr=.001)
    if kind=='bp_adam' and not hasattr(actor,'actor_optimizer'):actor.actor_optimizer=torch.optim.AdamW(actor.parameters(),lr=.003,weight_decay=0.)
    if kind not in ['bp','bp_adam'] and not hasattr(actor,'head_optimizer'):actor.head_optimizer=torch.optim.AdamW([v for n,v in actor.named_parameters() if not core_name(n)],lr=.003,weight_decay=0.)
    rng=np.random.default_rng(seed);pending=None;version=0;pending_version=None;rows=[];evaluations=[];start=time.monotonic();meta_updates=0;deep_updates=0
    for index,sequence in enumerate(blocks):
        tick=time.monotonic();t,m,mask=one_block(sequence);named=dict(actor.named_parameters());before=torch.cat([p.detach().flatten() for n,p in named.items() if core_name(n)])
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
                optimizer.zero_grad();(loss/.25).backward();torch.nn.utils.clip_grad_norm_(field.parameters(),5.);optimizer.step();meta_updates+=1;version+=1
            # All feedback on the old function version is consumed before issuing again.
            pending=None
            actor.head_optimizer.zero_grad();head_norm=torch.stack([g.square().sum() for g in heads.values()]).sum().clamp_min(1e-24).sqrt();head_scale=(5/head_norm).clamp_max(1)
            for name in head_names:named[name].grad=(head_scale*heads[name]).detach()
            actor.head_optimizer.step()
            with torch.set_grad_enabled(kind=='online'):pending=issue(actor,field,features,heads)
            pending_version=version;copy_issued(actor,pending)
        after=torch.cat([p.detach().flatten() for n,p in actor.named_parameters() if core_name(n)]);change=float((after-before).norm());deep_updates+=int(change>0)
        rows.append(dict(block=index,nll=float(loss.detach()),targets=16,teacher=teacher,deep_parameter_change=change,wall_s=time.monotonic()-tick))
        if evaluate is not None and ((index+1)%32==0 or index==len(blocks)-1):
            evaluation=evaluate(actor);evaluation.update(block=index+1,scored_targets=16*(index+1),elapsed_learning_s=time.monotonic()-start);evaluations.append(evaluation)
    # An unobserved final intervention has no target teaching it.
    actor.cut=False;pending=None
    return dict(wall_s=time.monotonic()-start,scored_targets=16*len(blocks),mean_nll=float(np.mean([r['nll'] for r in rows])),meta_updates=meta_updates,deep_updates=deep_updates,rows=rows,evaluations=evaluations),optimizer

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);ap.add_argument('--mode',choices=['smoke','fit'],required=True);ap.add_argument('--acquisition',type=int,choices=[128],required=True);ap.add_argument('--seed',type=int,choices=[1,2],required=True);a=ap.parse_args();out=ROOT/f'experiments/results/credit/{a.tag}.json';assert not out.exists();torch.set_num_threads(1);torch.set_default_dtype(torch.float64);start=time.monotonic();backend_tick=time.monotonic();extension();backend_startup=time.monotonic()-backend_tick
    if a.mode=='fit':
        gate=json.loads((ROOT/'experiments/results/credit/curie_native_optimized_credit_20261011_v1_smoke.json').read_text());assert gate['gate']
        for name,digest in gate['source_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    sequences=[s for s in native.load_split('taxi','train') if len(s[1])>=32];assert len(sequences)>=160
    gaps=np.concatenate([np.diff(t[:32]) for t,_ in sequences[:32]]);gaps=gaps[gaps>0];scale=float(np.median(gaps));qs=np.log(np.quantile(gaps,[.1,.5,.9])).tolist();K=1+max(int(m.max()) for _,m in sequences)
    acquisition,scored,depths=(4,8,[4]) if a.mode=='smoke' else (a.acquisition,512,[4,8]);offset=1328 if a.mode=='smoke' else 256;assert len(sequences)>=offset+acquisition+scored
    validation=[seq for seq in native.load_split('taxi','dev') if len(seq[1])>=32][:32];assert len(validation)==32
    parent_path=ROOT/'experiments/results/credit/curie_native_future_taught_20261011_v1_fit.json';parent=json.loads(parent_path.read_text());assert parent['status']=='completed'
    for name,digest in parent['source_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    cp_path=ROOT/parent['checkpoint'];assert hashlib.sha256(cp_path.read_bytes()).hexdigest()==parent['checkpoint_sha256'];reference=torch.load(cp_path,weights_only=True)
    def evaluate(actor):
        tick=time.monotonic();actor.cut=False
        with torch.no_grad():values=[float(actor(*one_block(sequence))) for sequence in validation]
        return dict(dev_nll=float(np.mean(values)),evaluation_wall_s=time.monotonic()-tick,scored_dev_targets=512)
    result_rows=[];checkpoints={}
    for depth in depths:
        model=initial(99910+a.seed,depth,K,scale,qs);start_actor=deepcopy(model.state_dict());model.load_state_dict(reference[f'{depth}/bp']['actor']);target=evaluate(model);strong_path=ROOT/'experiments/results/credit/curie_native_optimized_credit_20261011_v1_fit.json';strong=json.loads(strong_path.read_text());assert strong['status']=='completed';target['dev_nll']=next(row['arms']['bp_adam']['final_dev_nll'] for row in strong['rows'] if row['depth']==depth);target['source']='completed_seed0_optimized_native_AdamW';model.load_state_dict(start_actor);print(json.dumps(dict(depth=depth,reference_target=target)),flush=True);dimension=sum(p.numel() for n,p in model.named_parameters() if core_name(n));field=Credit((depth+2)*8+2,dimension)
        acquired,meta=learn(model,field,sequences[offset:offset+acquisition],'online',99911)
        shared_actor=deepcopy(model.state_dict());shared_field=deepcopy(field.state_dict());shared_meta=deepcopy(meta.state_dict());shared_head=deepcopy(model.head_optimizer.state_dict());arms={}
        for arm in ['online','frozen','bp','bp_adam']:
            actor=initial(99910+a.seed,depth,K,scale,qs)
            if arm in ['bp','bp_adam']:
                actor.load_state_dict(start_actor);cost,_=learn(actor,None,sequences[offset:offset+acquisition],arm,99911);credit=None;optimizer=None
            else:
                actor.load_state_dict(shared_actor);credit=Credit((depth+2)*8+2,dimension);credit.load_state_dict(shared_field);optimizer=torch.optim.Adam(credit.parameters(),lr=.001);optimizer.load_state_dict(shared_meta);actor.head_optimizer=torch.optim.AdamW([v for n,v in actor.named_parameters() if not core_name(n)],lr=.003,weight_decay=0.);actor.head_optimizer.load_state_dict(shared_head);cost=acquired
            initial_dev=evaluate(actor);measured,_=learn(actor,credit,sequences[offset+acquisition:offset+acquisition+scored],arm,99912,optimizer,evaluate);measured['acquisition_dev']=initial_dev;measured['reference_target']=target;reached=[v for v in measured['evaluations'] if v['dev_nll']<=target['dev_nll']];measured['work_to_reference_s']=cost['wall_s']+backend_startup if initial_dev['dev_nll']<=target['dev_nll'] else (None if not reached else cost['wall_s']+backend_startup+reached[0]['elapsed_learning_s']);measured['final_dev_nll']=measured['evaluations'][-1]['dev_nll'];measured['reference_quality_reached']=measured['work_to_reference_s'] is not None
            measured['acquisition']=cost;measured['whole_learning_wall_s']=cost['wall_s']+measured['wall_s']+backend_startup;measured['native_parameters']=sum(p.numel() for p in actor.parameters());measured['credit_parameters']=0 if credit is None else sum(p.numel() for p in credit.parameters());arms[arm]=measured
            checkpoints[f'{depth}/{arm}']=dict(actor=actor.state_dict(),credit=None if credit is None else credit.state_dict())
            print(json.dumps(dict(depth=depth,arm=arm,mean_nll=measured['mean_nll'],whole_learning_wall_s=measured['whole_learning_wall_s'],deep_updates=measured['deep_updates'],meta_updates=measured['meta_updates'],final_dev_nll=measured['final_dev_nll'],work_to_reference_s=measured['work_to_reference_s'])),flush=True)
        result_rows.append(dict(depth=depth,arms=arms,online_gain_vs_frozen=arms['frozen']['mean_nll']-arms['online']['mean_nll'],online_gain_vs_bp=arms['bp']['mean_nll']-arms['online']['mean_nll']))
    gate=all(r['arms']['online']['deep_updates']>0 and r['arms']['online']['meta_updates']>0 for r in result_rows);assert gate
    paths=['experiments/credit/run_native_optimized_credit.py','experiments/credit/NATIVE_OPTIMIZED_CREDIT_PROTOCOL.md','experiments/tpp/temporal_scan_cpu.py','experiments/tpp/temporal_scan_cpu.cpp','experiments/credit/run_native_future_taught_learning.py','experiments/credit/NATIVE_FUTURE_TAUGHT_LEARNING_PROTOCOL.md','experiments/credit/run_native_optimized_credit_confirmation.py','experiments/credit/NATIVE_OPTIMIZED_CREDIT_CONFIRMATION_PROTOCOL.md','experiments/tpp/race_tpp_v19.py'];cp=out.with_suffix('.pt');torch.save(checkpoints,cp)
    record=dict(strong_reference_result_sha256=hashlib.sha256(strong_path.read_bytes()).hexdigest(),backend='compiled_exact_native_temporal_scan',backend_startup_wall_s=backend_startup,parent_result_sha256=hashlib.sha256(parent_path.read_bytes()).hexdigest(),reference_checkpoint_sha256=parent['checkpoint_sha256'],validation_data_sha256=hashlib.sha256((ROOT/'data/easytpp/taxi/dev.json').read_bytes()).hexdigest(),status='completed',battle='B1',tag=a.tag,mode=a.mode,gate=gate,rows=result_rows,config=dict(seed=99910+a.seed,replicate_seed=a.seed,offset=offset,acquisition_blocks=acquisition,scored_blocks=scored,targets_per_block=16,history_per_block=16,teacher_probability=.25,actor_sgd_lr=.001,credit_adam_lr=.001,gradient_cap=5),wall_s=time.monotonic()-start,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},data_sha256=hashlib.sha256((ROOT/'data/easytpp/taxi/train.json').read_bytes()).hexdigest(),checkpoint=str(cp.relative_to(ROOT)),checkpoint_sha256=hashlib.sha256(cp.read_bytes()).hexdigest(),scope='First-pass native deep-parameter learning intervention, actual temporal likelihood and all model parameters adaptive, connected credit proposes all core gradients; next-block future likelihood teaches finite updates at audited p=.25. Exact local head gradients; native BP same actor/update frequency, common learned acquisition charged online/frozen. Private TRAIN sequences only, not chronological markets or official test. Dense internal temporal writes and deterministic mark addressing remain: no optional-route/counterfactual learning, full sparse substrate or supremacy claim. Fixed32sequence512targetTaxiDEV scoring, target chosen from completed native BP reference before new fits; recorded checkpoint wall-to-target is an upper bound on crossing time, unreached remains None. Acquisition horizons128/512 registered together,512scoring blocks; reused DEV selections are development only. CPU wall includes acquisition, forward, local backward, teacher, optimizer and repeated scoring; supported FLOPs not measured in this first diagnostic. All arms use the parity-validated compiled native temporal scan, first-order derivative only. Learned arms use exact local head AdamW.003 with inherited head optimizer state; deep generated updates remain SGD-scale.001. Native BP_SGD isolates old algorithm and BP_AdamW.003/wd0 is the stronger standard-native reference. All per-arm budgets allocate identical backend startup; teacher uses only first-order actor gradients, no actor Hessian. Independent initialization seeds1/2, same reused development data as seed0. Target fixed from completed seed0 optimized native AdamW, not tuned per repeat. Same scored data but acquisition updates differ across BP/learned arms.')
    out.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(dict(status=record['status'],gate=gate,wall_s=record['wall_s'])),flush=True)
if __name__=='__main__':main()
