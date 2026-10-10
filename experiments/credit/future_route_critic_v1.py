"""B1/R1 future intervention critic and learned candidate-pool development.
Frozen native temporal actor; controlled key-write interventions, not actor
weight training. Dense supervised continuations; no TEST or efficiency claim.
"""
import argparse
import hashlib
import json
import math
import resource
import time
from copy import deepcopy

import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode
from native_key_credit import ReceiverCuts, ROOT
from reciprocal_native_v1 import native_pair, batch

SOURCES=['experiments/credit/future_route_critic_v1.py','experiments/credit/native_key_credit.py',
         'experiments/credit/reciprocal_native_v1.py','experiments/tpp/recall_tpp_v4.py',
         'experiments/tpp/race_tpp_v5.py']


class InterventionModel(ReceiverCuts):
    intervention=None
    def encode(self,t,m,mask):
        h,slots=super().encode(t,m,mask)
        if self.intervention is not None:
            index,delta=self.intervention
            B,L,_=h.shape
            pulse=torch.nn.functional.one_hot(torch.tensor(index),L).to(h.dtype)
            keys=self.written_keys+pulse[None,:,None]*delta[:,None,:]
            state=h.new_zeros(B,self.K,self.dk);values=[]
            for j in range(L):
                onehot=torch.nn.functional.one_hot(m[:,j],self.K).to(h.dtype)
                state=state*self.key_decay[:,j,None,:]+onehot[:,:,None]*keys[:,j,None,:]
                values.append(state)
            self._keyed=torch.stack(values,1)
        return h,slots


def actor(depth,seed):
    original,_=native_pair(depth,seed)
    model=InterventionModel(8,8,3,depth,2,2,2,0.,1.,[-1.,0.],dk=4,
                            local=True,prev_msg=True,qk_norm=False).double()
    model.load_state_dict(original.model.state_dict())
    return model


def event_losses(model,t,m,index=None,delta=None):
    model.intervention=None if index is None else (index,delta)
    try:
        tl,ml,n,cache=model.loglik(t,m,torch.ones_like(m,dtype=torch.bool))
        lh,ls=model.clock_terms(cache[3],*cache[2][:4])
        selected=cache[2][4].gather(-1,m[:,1:,None,None].expand(-1,-1,model.M,1)).squeeze(-1)
        losses=-(torch.logsumexp(lh+selected,-1)+ls.sum(-1))
        return losses,cache
    finally:
        model.intervention=None


def prefix_packet(model,t,m,index):
    # Recompute only the revealed prefix: no future observations in features.
    losses,cache=event_losses(model,t[:,:index+1],m[:,:index+1])
    h=cache[0];q=model._query;k=model.written_keys
    previous=torch.cat((torch.zeros_like(h[:,:1]),model.embed(m[:,:index])),1)
    addressed=model._keyed.gather(2,m[:,:index+1,None,None].expand(-1,-1,1,4)).squeeze(2)
    dt=torch.diff(t[:,:index+1],prepend=t[:,:1],dim=1).clamp_min(0).log1p().unsqueeze(-1)
    observed=torch.cat((losses.new_zeros(len(t),1),losses),1).unsqueeze(-1)
    return torch.cat((h,q,k,previous,addressed,dt,observed),-1).detach()


def actions(radius=.25):
    eye=torch.eye(4,dtype=torch.float64)*radius
    return torch.cat((torch.zeros(1,4,dtype=torch.float64),eye,-eye),0)


class Critic(nn.Module):
    def __init__(self,connected):
        super().__init__();self.connected=connected
        self.register_buffer('output_scale',torch.tensor(1.,dtype=torch.float64))
        self.encoder=nn.GRU(30,32,batch_first=True) if connected else nn.Sequential(nn.Linear(30,96),nn.Tanh(),nn.Linear(96,32))
        self.head=nn.Sequential(nn.Linear(36,32),nn.Tanh(),nn.Linear(32,1))
    def context(self,x):
        x=x.tanh()
        return self.encoder(x)[0][:,-1] if self.connected else self.encoder(x[:,-1])
    def values(self,x,a):
        c=self.context(x);N,K,_=a.shape
        c=c[:,None,:].expand(-1,K,-1)
        return self.output_scale*(self.head(torch.cat((c,a),-1))-self.head(torch.cat((c,torch.zeros_like(a)),-1))).squeeze(-1)


def collect(model,start,count,pairs):
    packets=[];targets=[];immediate=[];records=[];work=0
    candidates=actions();index=2*pairs-1
    with torch.no_grad():
        for i in range(count):
            t,m=batch(start+i,size=1,pairs=pairs)
            packets.append(prefix_packet(model,t,m,index))
            base,_=event_losses(model,t,m)
            values=[];first=[]
            for a in candidates:
                changed,_=event_losses(model,t,m,index,a[None,:])
                difference=changed[:,index:]-base[:,index:]
                values.append(difference.mean());first.append(difference[:,0].mean())
            targets.append(torch.stack(values));immediate.append(torch.stack(first))
            records.append((t,m,index));work+=(len(candidates)+1)*(m.numel()-1)+index
    return torch.cat(packets),torch.stack(targets),torch.stack(immediate),records,work


def summarize(critic,x,target,initial):
    candidates=actions()[None,:,:].expand(len(x),-1,-1)
    with torch.no_grad():
        pred=critic.values(x,candidates);frozen=initial.values(x,candidates)
        selected=pred.argmin(-1);chosen=target.gather(1,selected[:,None]).squeeze(1)
        energy=target.square().sum().clamp_min(1e-30)
        return dict(relative_mse=float((pred-target).square().sum()/energy),
                    frozen_relative_mse=float((frozen-target).square().sum()/energy),
                    zero_prediction_relative_mse=1.,selected_future_advantage=float(chosen.mean()),
                    oracle_in_fixed_pool=float(target.min(-1).values.mean()),
                    uniform_pool_future_advantage=float(target.mean()),
                    chosen_advantages=chosen.tolist(),squared_error_by_packet=(pred-target).square().mean(-1).tolist())


def fit(model,seed,a):
    x,y,first,records,work=collect(model,100000+seed*100,a.train_packets,a.pairs)
    dx,dy,dfirst,drecords,dwork=collect(model,200000+seed*100,a.dev_packets,a.pairs)
    rows=[];histories={};candidates=actions()[None,:,:].expand(len(x),-1,-1)
    scale=float(y.square().mean().sqrt().clamp_min(1e-8))
    for connected in (False,True):
        torch.manual_seed(seed);critic=Critic(connected).double()
        critic.output_scale.fill_(scale);initial=deepcopy(critic)
        opt=torch.optim.AdamW(critic.parameters(),lr=.003,weight_decay=.001)
        history=[]
        for step in range(a.steps):
            prediction=critic.values(x,candidates)
            loss=((prediction-y)/scale).square().mean()
            opt.zero_grad();loss.backward();opt.step()
            if not torch.isfinite(loss):raise ValueError('Nonfinite critic fit')
            history.append(float(loss.detach()))
        name='connected' if connected else 'isolated'
        row=dict(arm=name,critic_parameters=sum(p.numel() for p in critic.parameters()),
                 train=summarize(critic,x,y,initial),dev=summarize(critic,dx,dy,initial))
        if connected:
            # One learned candidate augments, rather than removes, audited basis routes.
            for p in critic.parameters():p.requires_grad_(False)
            torch.manual_seed(seed+1);proposal=nn.Sequential(nn.Linear(32,32),nn.Tanh(),nn.Linear(32,4)).double()
            popt=torch.optim.AdamW(proposal.parameters(),lr=.003,weight_decay=.001)
            context=critic.context(x).detach()
            def propose(c):
                v=proposal(c);return .25*v/(1+v.square().sum(-1,keepdim=True).sqrt())
            for step in range(a.proposal_steps):
                delta=propose(context)
                loss=critic.values(x,delta[:,None,:]).mean()/scale
                popt.zero_grad();loss.backward();popt.step()
                if not torch.isfinite(loss):raise ValueError('Nonfinite proposal fit')
            with torch.no_grad():
                delta=propose(critic.context(dx));proposal_values=[]
                for i,(t,m,index) in enumerate(drecords):
                    base,_=event_losses(model,t,m)
                    changed,_=event_losses(model,t,m,index,delta[i:i+1])
                    proposal_values.append((changed[:,index:]-base[:,index:]).mean())
                    dwork+=2*(m.numel()-1)
                actual=torch.cat((dy,torch.stack(proposal_values)[:,None]),1)
                pool=torch.cat((actions()[None,:,:].expand(len(dx),-1,-1),delta[:,None,:]),1)
                pred=critic.values(dx,pool);selected=pred.argmin(-1)
                chosen=actual.gather(1,selected[:,None]).squeeze(1)
                fixed_oracle=dy.min(-1).values;augmented_oracle=actual.min(-1).values
                row['learned_pool']=dict(proposal_parameters=sum(p.numel() for p in proposal.parameters()),
                    radius=.25,fixed_candidates=9,augmented_candidates=10,
                    proposal_realized_advantages=torch.stack(proposal_values).tolist(),
                    selected_future_advantage=float(chosen.mean()),chosen_advantages=chosen.tolist(),
                    mean_pool_coverage_improvement=float((fixed_oracle-augmented_oracle).mean()),
                    mean_selection_regret=float((chosen-augmented_oracle).mean()),
                    scope='Bounded critic-trained proposal; actual held-out branches check critic exploitation. Basis candidates retained. Extra discovery/evaluation charged.')
        rows.append(row);histories[name]=history
    return dict(arms=rows,training_histories=histories,target_scale_train_only=scale,
                teacher_scored_targets=work+dwork,train_packets=a.train_packets,dev_packets=a.dev_packets,
                immediate_vs_future_best_action_disagreement=float((first.argmin(-1)!=y.argmin(-1)).double().mean()),
                scope='Frozen initialized native actor; interventions in one addressed key write, not parameter learning or asynchronous scheduling. Connected/isolation parameter counts differ; no isolated connectivity or efficiency claim. Future labels supervise fitting only; DEV candidates selected before branch labels are consulted.')


def contracts():
    rows=[]
    for depth in (2,4,8):
        model=actor(depth,173);t,m=batch(173,size=1,pairs=2);j=3
        base,_=event_losses(model,t,m);original,_=native_pair(depth,173)
        assert abs(float(base.mean()-original(t,m)))<1e-12
        zero,_=event_losses(model,t,m,j,torch.zeros(1,4));assert torch.equal(base,zero)
        packet=prefix_packet(model,t,m,j)
        changed_m=m.clone();changed_m[:,j+1:]=(changed_m[:,j+1:]+1)%8
        changed_t=t.clone();changed_t[:,j+1:]+=10
        assert torch.equal(packet,prefix_packet(model,changed_t,changed_m,j))
        delta=torch.zeros(1,4,requires_grad=True,dtype=torch.float64)
        intervention,_=event_losses(model,t,m,j,delta)
        grad=torch.autograd.grad(intervention[:,j:].mean(),delta)[0]
        v=torch.tensor([[.1,-.2,.3,-.4]],dtype=torch.float64);eps=1e-5
        plus,_=event_losses(model,t,m,j,eps*v);minus,_=event_losses(model,t,m,j,-eps*v)
        fd=float((plus[:,j:].mean()-minus[:,j:].mean())/(2*eps))
        error=abs(fd-float((grad*v).sum()));assert error<1e-8,error
        moved,_=event_losses(model,t,m,j,.1*v)
        assert torch.equal(base[:,:j],moved[:,:j])
        rows.append(dict(depth=depth,zero_intervention_parity=True,prefix_causality=True,
                         past_loss_unchanged=True,directional_derivative_error=error))
    return rows


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True)
    ap.add_argument('--mode',choices=['contract','smoke','pilot'],required=True)
    ap.add_argument('--seed',type=int,default=170);ap.add_argument('--depth',type=int,default=2)
    ap.add_argument('--pairs',type=int,default=2);ap.add_argument('--steps',type=int,default=128)
    ap.add_argument('--proposal-steps',type=int,default=32);ap.add_argument('--train-packets',type=int,default=16)
    ap.add_argument('--dev-packets',type=int,default=16);a=ap.parse_args()
    out=ROOT/'experiments/results/credit'/(a.tag+'.json')
    if out.exists():raise FileExistsError(out)
    if a.mode=='smoke':a.steps=2;a.proposal_steps=2;a.train_packets=2;a.dev_packets=2
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64);start=time.monotonic()
    with FlopCounterMode(display=False) as counter:
        result=contracts() if a.mode=='contract' else fit(actor(a.depth,a.seed),a.seed,a)
    record=dict(status='completed',tag=a.tag,battle='R1/B1',args=vars(a),training=a.mode!='contract',
                decision='Whether connected dense-feedback future credit learns held-out intervention value and a jointly trained candidate improves coverage',
                metrics=result,wall_s=time.monotonic()-start,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                supported_flops=counter.get_total_flops(),flop_scope='Supported PyTorch formulas; special/unsupported operations are excluded, wall includes profiling. All branch and critic/proposal work included.',
                source_sha256={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in SOURCES})
    tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(record,indent=2)+'\n');tmp.replace(out)
    print('RESULT',json.dumps({k:v for k,v in record.items() if k!='metrics'}),flush=True)


if __name__=='__main__':main()
