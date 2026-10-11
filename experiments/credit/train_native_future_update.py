"""B1 full-native future-taught update control; all lookahead work charged."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import resource
import sys
import time
import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/tpp'))
import race_tpp_v19 as native
SOURCES=['experiments/credit/train_native_future_update.py','experiments/tpp/race_tpp_v19.py']

class Actor(native.RaceTPP):
    def forward(self,t,m,mask):
        tl,ml,count,_=self.loglik(t,m,mask)
        return -(tl+ml)/count,tl,ml,count

class Controller(torch.nn.Module):
    def __init__(self):
        super().__init__();self.net=torch.nn.Sequential(torch.nn.Linear(6,16),torch.nn.Tanh(),torch.nn.Linear(16,4))
        with torch.no_grad():self.net[-1].weight.zero_();self.net[-1].bias.fill_(-math.log(4))
    def forward(self,features):return .25+3.75*torch.sigmoid(self.net(features))

def group(name):
    if name.startswith(('embed.','gap.')):return 0
    if name.startswith('layers.'):return 1
    if name.startswith('clock.'):return 2
    return 3

def batch(seqs):
    return (torch.tensor(np.stack([s[0] for s in seqs])),torch.tensor(np.stack([s[1] for s in seqs])),torch.ones(len(seqs),len(seqs[0][0]),dtype=torch.bool))

def score(actor,seqs):
    tl=ml=count=0.
    with torch.no_grad():
        for i in range(0,len(seqs),8):
            _,t,m,c=actor(*batch(seqs[i:i+8]));tl+=float(t);ml+=float(m);count+=int(c)
    return dict(nll=-(tl+ml)/count,time_nll=-tl/count,mark_nll=-ml/count,targets=count)

def fit(args,depth,train,dev,scale,quantiles,K):
    torch.manual_seed(args.seed+3000+depth)
    original=Actor(K,16,8,depth,2,4,4,0.,scale,quantiles).double()
    initial={k:v.detach().clone() for k,v in original.state_dict().items()}
    rows=[];weights={}
    for arm in ['fixed_adam','future_taught']:
        actor=Actor(K,16,8,depth,2,4,4,0.,scale,quantiles).double();actor.load_state_dict(initial)
        torch.manual_seed(args.seed+3100+depth);controller=Controller().double();meta=torch.optim.Adam(controller.parameters(),lr=.001)
        names=list(dict(actor.named_parameters()));groups=[group(n) for n in names]
        moments={n:(torch.zeros_like(p),torch.zeros_like(p)) for n,p in actor.named_parameters()}
        history=[];updates=0;presentations=0;lookahead_targets=0;fit_s=eval_s=0.;supported_flops=0;meta_steps=0
        for epoch in range(args.epochs):
            order=np.random.default_rng(args.seed+epoch).permutation(len(train))
            ordered=[train[i] for i in order];blocks=[batch(ordered[i:i+8]) for i in range(0,len(ordered),8)]
            tick=time.monotonic()
            with FlopCounterMode(display=False) as counter:
                for i,data in enumerate(blocks):
                    parameters=dict(actor.named_parameters());actor.zero_grad()
                    loss,_,_,count=actor(*data);assert torch.isfinite(loss)
                    gradients=torch.autograd.grad(loss,list(parameters.values()))
                    norm=torch.sqrt(sum(g.detach().square().sum() for g in gradients));clip=(10/norm.clamp_min(1e-30)).clamp_max(1)
                    gradients=[g.detach()*clip for g in gradients];updates+=1;presentations+=int(count)
                    directions={};group_norm=[]
                    for n,g in zip(names,gradients):
                        m,v=moments[n];m=.9*m+.1*g;v=.999*v+.001*g.square();moments[n]=(m,v)
                        directions[n]=(m/(1-.9**updates))/(torch.sqrt(v/(1-.999**updates))+1e-8)
                    for index in range(4):
                        values=[g.square().mean() for g,k in zip(gradients,groups) if k==index]
                        group_norm.append(torch.log1p(torch.stack(values).mean().sqrt()))
                    features=torch.stack([loss.detach().clamp(-100,100)/10,torch.tensor(epoch/args.epochs),*group_norm]).detach()
                    factors=controller(features) if arm=='future_taught' else torch.ones(4)
                    proposed={n:parameters[n].detach()-.001*factors[group(n)]*directions[n] for n in names}
                    # Issue the actor update before receiving the following batch's feedback.
                    with torch.no_grad():
                        for n,p in parameters.items():p.copy_(proposed[n])
                    if arm=='future_taught' and i+1<len(blocks):
                        future,_,_,future_count=torch.func.functional_call(actor,proposed,blocks[i+1])
                        assert torch.isfinite(future)
                        objective=future+.001*torch.log(factors).square().mean()
                        meta.zero_grad();objective.backward();torch.nn.utils.clip_grad_norm_(controller.parameters(),1.);meta.step()
                        meta_steps+=1;lookahead_targets+=int(future_count)
            fit_s+=time.monotonic()-tick;supported_flops+=counter.get_total_flops()
            tick=time.monotonic();quality=score(actor,dev);eval_s+=time.monotonic()-tick
            history.append(dict(epoch=epoch+1,**quality,fit_wall_s=fit_s,supported_fit_flops=supported_flops,
                                factors=factors.detach().tolist()))
        rows.append(dict(depth=depth,arm=arm,history=history,fit_wall_s=fit_s,evaluation_wall_s=eval_s,
                         supported_fit_flops=supported_flops,updates=updates,train_target_presentations=presentations,
                         extra_lookahead_targets=lookahead_targets,controller_updates=meta_steps,
                         actor_parameters=sum(p.numel() for p in actor.parameters()),controller_parameters=sum(p.numel() for p in controller.parameters())))
        weights[arm]=dict(actor=actor.state_dict(),controller=controller.state_dict(),meta=meta.state_dict(),moments=moments)
    return rows,weights

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--tag',required=True);ap.add_argument('--stage',choices=['smoke','fit'],required=True)
    ap.add_argument('--seed',type=int,default=0);ap.add_argument('--gate');args=ap.parse_args()
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64);start=time.monotonic()
    args.epochs=1 if args.stage=='smoke' else 3
    count=16 if args.stage=='smoke' else 2048;length=8 if args.stage=='smoke' else 32
    depths=[2] if args.stage=='smoke' else [2,4,8]
    inputs={}
    if args.stage=='fit':
        assert args.gate;path=ROOT/args.gate;gate=json.loads(path.read_text());assert gate['status']=='completed' and gate['stage']=='smoke'
        for s,digest in gate['source_sha256'].items():assert hashlib.sha256((ROOT/s).read_bytes()).hexdigest()==digest
        inputs[args.gate]=hashlib.sha256(path.read_bytes()).hexdigest()
    raw=native.load_split('taxi','train');validation=native.load_split('taxi','dev')
    K=1+max(int(m.max()) for _,m in raw)
    eligible=[(t[:length],m[:length]) for t,m in raw if len(m)>=length];assert len(eligible)>=count
    train=eligible[:count];dev=[(t[:length],m[:length]) for t,m in validation if len(m)>=length][:8 if args.stage=='smoke' else 128]
    gaps=np.concatenate([np.diff(t) for t,_ in train]);positive=gaps[gaps>0];scale=float(np.median(positive));quantiles=np.log(np.quantile(positive,[.1,.35,.65,.9])).tolist()
    rows=[];weights={}
    for depth in depths:
        result,saved=fit(args,depth,train,dev,scale,quantiles,K);rows+=result;weights[str(depth)]=saved
    folder=ROOT/'experiments/results/credit';checkpoint=folder/f'{args.tag}.pt';assert not checkpoint.exists()
    torch.save(weights,checkpoint)
    output=dict(status='completed',tag=args.tag,battle='B1 native future-taught learning',stage=args.stage,rows=rows,
                args=vars(args),train_sequences=count,events_per_sequence=length,dev_sequences=len(dev),
                final_weights=str(checkpoint.relative_to(ROOT)),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                source_sha256={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in SOURCES},input_sha256=inputs,
                data_sha256={s:hashlib.sha256((ROOT/f'data/easytpp/taxi/{s}.json').read_bytes()).hexdigest() for s in ['train','dev']},
                wall_s=time.monotonic()-start,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                scope='Matched native models/all parameters updated, fixed Adam versus future-taught bounded group scales. Independent sequences carry temporal/addressed state within32events; dense writes, no hard-route or cross-sequence carried learner credit. Every extra TRAIN feedback pass/controller backward/moment update charged; supported FLOPs omit uncounted operations. DEV only, three-pass acquisition stage, no Transformer/benchmark supremacy claim.')
    with (folder/f'{args.tag}.json').open('x') as f:json.dump(output,f,indent=2)
    print(json.dumps(output))

if __name__=='__main__':main()
