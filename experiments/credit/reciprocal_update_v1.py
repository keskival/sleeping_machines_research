"""B1/R1 shared enabler: learn sparse temporal model updates across episodes.

Exact inner differentiation is retained. The experiment tests update quality,
not replacement of credit inference or elimination of meta-training gradients.
Every numerical contract, smoke and fit runs through a battle's safe queue.
"""
import argparse
import hashlib
import json
import math
import resource
import socket
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
K, C, D, Q = 4, 4, 2, 4
BLOCKS = ('value', 'key', 'query', 'readout', 'bias', 'log_rate', 'frequency')
DT = torch.float64


def initial(seed=1234):
    gen = torch.Generator().manual_seed(seed)
    rand = lambda *s: torch.randn(*s, generator=gen, dtype=DT)
    return dict(value=rand(C,2*D)*.3, key=torch.eye(K,dtype=DT)+rand(K,K)*.05,
                query=torch.eye(K,dtype=DT)+rand(K,K)*.05, readout=rand(2*D,2)*.3,
                bias=torch.zeros(2,dtype=DT), log_rate=torch.full((D,),-4.,dtype=DT),
                frequency=torch.tensor([.025,.06],dtype=DT))


def episode(gen, labels, batch):
    # Four observed key/value writes, then four queries WITH replacement.
    # Value identities are inputs; task-specific output marks appear only in
    # observed support losses or outer evaluation, never the causal router.
    values=torch.randint(C,(batch,K),generator=gen)
    gaps=torch.exp(torch.rand(batch,K,generator=gen,dtype=DT)*3-1)
    stamps=gaps.cumsum(-1)
    query=torch.randint(K,(batch,Q),generator=gen)
    query_times=stamps[:,-1,None]+torch.exp(torch.rand(batch,Q,generator=gen,dtype=DT)*3-1).cumsum(-1)
    target=labels[values.gather(1,query)]
    # Teacher is a marked exponential race: total rate 2, desired mark .8.
    flip=torch.rand(batch,Q,generator=gen)<.2
    marks=torch.where(flip,1-target,target)
    waits=-torch.log(torch.rand(batch,Q,generator=gen,dtype=DT).clamp_min(1e-12))/2
    return dict(values=values,stamps=stamps,query=query,times=query_times,marks=marks,waits=waits)


def task(seed,batch):
    gen=torch.Generator().manual_seed(seed)
    labels=torch.randint(2,(C,),generator=gen)
    return episode(gen,labels,batch),episode(gen,labels,batch),episode(gen,labels,batch)


def scores(p,e):
    # Sparse addressed writes: each observed key writes exactly one slot.
    # Query reads all keys, while a sampled route would deliver one value.
    # Exact likelihood training enumerates four hidden read causes and charges
    # every value delivery. Deterministic expected-value reads are NOT used.
    content=p['value'][e['values']]
    elapsed=e['times'][:,:,None]-e['stamps'][:,None,:]
    angle=elapsed[...,None]*p['frequency']
    retention=torch.exp(-elapsed[...,None]*p['log_rate'].exp())
    real,imag=content[:,:,:D],content[:,:,D:]
    zr=retention*(angle.cos()*real[:,None]-angle.sin()*imag[:,None])
    zi=retention*(angle.sin()*real[:,None]+angle.cos()*imag[:,None])
    memory=torch.cat((zr,zi),-1)
    keys=F.normalize(p['key'],dim=-1)
    queries=F.normalize(p['query'][e['query']],dim=-1)
    logprior=F.log_softmax(3*(queries@keys.T),-1)
    logits=memory@p['readout']+p['bias']
    # Bounded rate parameterization, exact hazard/compensator for each cause.
    lograte=2*torch.tanh(logits/2)
    exposure=lograte.exp().sum(-1)*e['waits'][:,:,None]
    chosen=lograte.gather(-1,e['marks'][:,:,None,None].expand(-1,-1,K,1)).squeeze(-1)
    complete=logprior+chosen-exposure
    return complete,logprior,lograte


def loss(p,e):
    complete,_,_=scores(p,e)
    return -torch.logsumexp(complete,-1).mean()


def selected_rates(p,e,route):
    """Winner-only value delivery given a sampled/forced route; all-key discovery.

    This is the conditional forward path, not a claim that likelihood
    evaluation or complete learning performs just one value read.
    """
    values=e['values'].gather(1,route)
    stamp=e['stamps'].gather(1,route)
    elapsed=e['times']-stamp
    content=p['value'][values];real,imag=content[...,:D],content[...,D:]
    angle=elapsed[...,None]*p['frequency']
    retention=torch.exp(-elapsed[...,None]*p['log_rate'].exp())
    zr=retention*(angle.cos()*real-angle.sin()*imag)
    zi=retention*(angle.sin()*real+angle.cos()*imag)
    logits=torch.cat((zr,zi),-1)@p['readout']+p['bias']
    return 2*torch.tanh(logits/2)


class Update(torch.nn.Module):
    """Shared coordinate rule with persistent first/second moment memory.

Six local features + block identity; predicts a bounded positive gain on an
RMS-normalized credit direction. No full parameter-sized network per task.
"""
    def __init__(self,scalar=False):
        super().__init__();self.scalar=scalar
        if scalar:self.logit=torch.nn.Parameter(torch.tensor(math.log(.08/.42),dtype=DT))
        else:
            self.net=torch.nn.Sequential(torch.nn.Linear(6+len(BLOCKS),8),torch.nn.Tanh(),torch.nn.Linear(8,1)).to(DT)
            with torch.no_grad():
                self.net[-1].weight.zero_();self.net[-1].bias.fill_(math.log(.08/.42))

    def step(self,p,grads,state,step,nsteps,reset=False):
        updated={};nextstate={}
        for bi,(name,g) in enumerate(zip(BLOCKS,grads)):
            oldm,oldv=state.get(name,(torch.zeros_like(g),torch.zeros_like(g)))
            if reset:oldm,oldv=torch.zeros_like(g),torch.zeros_like(g)
            m=.8*oldm+.2*g;v=.9*oldv+.1*g.square()
            scale=torch.sqrt(v+1e-8)+.1
            if self.scalar:raw=self.logit.expand_as(g)
            else:
                tag=torch.zeros(g.numel(),len(BLOCKS),dtype=DT);tag[:,bi]=1
                feat=torch.stack((g/scale,m/scale,torch.log1p(v),p[name].tanh(),
                                  torch.full_like(g,(step+1)/nsteps),torch.full_like(g,1. if step==0 else 0.)),dim=-1)
                raw=self.net(torch.cat((feat.reshape(-1,6),tag),-1)).reshape_as(g)
            gain=.02+.48*raw.sigmoid()
            updated[name]=p[name]-gain*g/scale
            nextstate[name]=(m,v)
        return updated,nextstate


def adapt(p,support,rule,steps,lr=.1,full=False,reset=False):
    p={k:v.clone().requires_grad_(True) for k,v in p.items()}
    state={};history=[]
    for step in range(steps):
        l=loss(p,support);history.append(float(l.detach()))
        grads=torch.autograd.grad(l,tuple(p.values()),create_graph=full)
        if isinstance(rule,Update):p,state=rule.step(p,grads,state,step,steps,reset)
        elif rule=='sgd':p={k:v-lr*g for (k,v),g in zip(p.items(),grads)}
        elif rule=='adam':
            new={}
            for (k,v),g in zip(p.items(),grads):
                m,s=state.get(k,(torch.zeros_like(g),torch.zeros_like(g)))
                m=.9*m+.1*g;s=.999*s+.001*g.square();state[k]=(m,s)
                new[k]=v-lr*(m/(1-.9**(step+1)))/(torch.sqrt(s/(1-.999**(step+1)))+1e-8)
            p=new
        else:raise ValueError(rule)
        if not full:
            p={k:v.detach().requires_grad_(True) for k,v in p.items()}
            state={k:tuple(t.detach() for t in v) for k,v in state.items()}
    return p,history


def evaluate(base,learner,a):
    # Selection tasks and assessment tasks are disjoint synthetic DEV pools.
    # No public benchmark TEST, no data-dependent change to initialization.
    rows=[]
    for pool,start in [('selection',200000),('assessment',300000)]:
        candidates=[('no_update',None,0.)]
        candidates += [(kind,kind,lr) for kind in ('sgd','adam') for lr in (.01,.03,.1,.3)]
        candidates += [('learned',learner,0.),('learned_reset',learner,0.)]
        for kind,rule,lr in candidates:
            total=0.;initial_total=0.;gaps=[];start_time=time.time()
            for i in range(a.dev_tasks):
                support,_,query=task(start+i,a.batch)
                initial_loss=float(loss(base,query))
                adapted=base if rule is None else adapt(base,support,rule,a.inner_steps,lr,reset=kind=='learned_reset')[0]
                value=float(loss(adapted,query));total+=value;initial_total+=initial_loss;gaps.append(value-initial_loss)
            rows.append(dict(pool=pool,arm=kind,lr=lr,mean_query_nll=total/a.dev_tasks,
                             mean_initial_nll=initial_total/a.dev_tasks,paired_change=gaps,
                             wall_s=time.time()-start_time,
                             adaptation_gradient_targets=0 if rule is None else a.dev_tasks*a.inner_steps*a.batch*Q,
                             assessed_targets=a.dev_tasks*a.batch*Q,
                             value_causes_per_likelihood_target=K))
    selected={kind:min((r for r in rows if r['pool']=='selection' and r['arm']==kind),key=lambda r:r['mean_query_nll'])['lr'] for kind in ('sgd','adam')}
    comparison=[r for r in rows if r['pool']=='assessment' and (r['arm'] not in selected or r['lr']==selected[r['arm']])]
    return dict(rows=rows,selected_baseline_lr=selected,assessment=comparison)


def run(a):
    torch.manual_seed(a.seed);base=initial(a.model_seed)
    learner=Update(scalar=a.arm=='scalar');opt=torch.optim.Adam(learner.parameters(),lr=.003)
    history=[];start=time.time();counts=dict(inner_gradient_targets=0,outer_targets=0,meta_updates=0)
    for outer in range(a.meta_steps):
        # Identical task stream for all arms; seed varies learner only.
        support,future,_=task(a.data_seed+outer,a.batch)
        p,_=adapt(base,support,learner,a.inner_steps,full=True)
        objective=loss(p,support if a.arm=='immediate' else future)
        assert torch.isfinite(objective),'Nonfinite meta objective'
        opt.zero_grad();objective.backward();torch.nn.utils.clip_grad_norm_(learner.parameters(),1.);opt.step()
        counts['inner_gradient_targets']+=a.batch*Q*a.inner_steps
        counts['outer_targets']+=a.batch*Q;counts['meta_updates']+=1
        row=dict(step=outer,outer_nll=float(objective.detach()))
        history.append(row)
        if outer%10==0:print(json.dumps(row),flush=True)
    meta_wall_s=time.time()-start
    dev=evaluate(base,learner,a)
    return dict(status='completed',args=vars(a),history=history,evaluation=dev,wall_s=time.time()-start,
                base_parameters=sum(v.numel() for v in base.values()),backward_parameters=sum(v.numel() for v in learner.parameters()),
                work_counts=counts,work_scope='Counts are targets, not FLOPs. Full differentiable inner unroll includes higher-order meta work; all four causes evaluated per target; evaluation and meta-training overhead must be charged.',
                meta_training_wall_s=meta_wall_s,
                mechanism_scope='One temporal addressed key/value memory, decay/rotation, stochastic hard read routing with exact losing-route likelihood credit, marked event/silence likelihood. Known input-key writes; no learned write choice, deep memory or learned credit inference.',
                selection='Fixed final meta step; baseline step sizes selected on separate synthetic DEV selection tasks; assessment is development, not benchmark TEST',
                model_sha256=digest_tensors(base),training_data_sha256=data_digest(a),
                update_sha256=digest_tensors(learner.state_dict()),
                evidence_level='first-pass learning-rule development diagnostic')


def digest_tensors(values):
    h=hashlib.sha256()
    for k,v in values.items():h.update(k.encode());h.update(v.detach().contiguous().numpy().tobytes())
    return h.hexdigest()


def data_digest(a):
    h=hashlib.sha256()
    for i in range(a.meta_steps):
        for e in task(a.data_seed+i,a.batch):
            for k,v in e.items():h.update(k.encode());h.update(v.numpy().tobytes())
    return h.hexdigest()


def contracts(a):
    torch.manual_seed(7);base=initial();support,future,_=task(5,2)
    p={k:v.clone().requires_grad_(True) for k,v in base.items()}
    complete,_,_=scores(p,support)
    direct=torch.autograd.grad(-torch.logsumexp(complete,-1).mean(),tuple(p.values()))
    complete,_,_=scores(p,support);rho=complete.softmax(-1).detach()
    posterior_grad=torch.autograd.grad(-(rho*complete).sum(-1).mean(),tuple(p.values()))
    error=max((x-y).abs().max().item() for x,y in zip(direct,posterior_grad));assert error<1e-12
    # Full outer derivative, including changed inner gradients, vs finite difference.
    learner=Update(scalar=True)
    outer=lambda:loss(adapt(base,support,learner,3,full=True)[0],future)
    meta_grad=torch.autograd.grad(outer(),learner.logit)[0].item()
    original=learner.logit.item();eps=1e-5
    with torch.no_grad():learner.logit.fill_(original+eps)
    plus=float(outer())
    with torch.no_grad():learner.logit.fill_(original-eps)
    minus=float(outer())
    with torch.no_grad():learner.logit.fill_(original)
    derivative_error=abs(meta_grad-(plus-minus)/(2*eps));assert derivative_error<1e-7
    # Targets cannot change causal memory/routing or pre-outcome rates.
    altered={k:v.clone() for k,v in support.items()};altered['marks']=1-altered['marks'];altered['waits']*=2
    _,prior,rates=scores(base,support);_,prior2,rates2=scores(base,altered)
    assert torch.equal(prior,prior2) and torch.equal(rates,rates2)
    # Nonzero temporal effect; simultaneous rotation/decay transport semigroup.
    elapsed=.7;later=1.2;r=base['log_rate'].exp();w=base['frequency']
    transport=lambda t:torch.exp(torch.complex(-r*t,w*t))
    transport_error=(transport(elapsed)*transport(later)-transport(elapsed+later)).abs().max().item();assert transport_error<1e-12
    shifted={k:v.clone() for k,v in support.items()};shifted['times']+=5
    temporal_effect=(scores(base,shifted)[2]-rates).abs().max().item();assert temporal_effect>1e-4
    routes=torch.randint(K,support['query'].shape,generator=torch.Generator().manual_seed(9))
    selected=selected_rates(base,support,routes)
    selected_error=(selected-rates.gather(2,routes[:,:,None,None].expand(-1,-1,1,2)).squeeze(2)).abs().max().item()
    assert selected_error<1e-12
    # Different future evidence changes update learner's training signal.
    alternate={k:v.clone() for k,v in future.items()};alternate['marks']=1-alternate['marks']
    p,_=adapt(base,support,learner,3,full=True)
    g1=torch.autograd.grad(loss(p,future),learner.logit,retain_graph=True)[0]
    g2=torch.autograd.grad(loss(p,alternate),learner.logit)[0]
    assert abs(float(g1-g2))>1e-6
    return dict(status='completed',checks=dict(all_parameter_posterior_gradient_error=error,
                outer_meta_gradient_finite_difference_error=derivative_error,causal_target_independence=True,
                transport_semigroup_error=transport_error,elapsed_time_effect=temporal_effect,
                selected_value_path_error=selected_error,
                future_evidence_meta_signal_difference=float(g1-g2)),scope='Numerical contracts only; no fitted quality')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True)
    ap.add_argument('--arm',choices=['future','immediate','scalar'],default='future');ap.add_argument('--contract',action='store_true')
    for name,default in [('seed',0),('model-seed',1234),('data-seed',40000),('meta-steps',80),('inner-steps',4),('batch',8),('dev-tasks',16)]:ap.add_argument('--'+name,type=int,default=default)
    a=ap.parse_args();assert min(a.meta_steps,a.inner_steps,a.batch,a.dev_tasks)>0
    torch.set_num_threads(1)
    out=ROOT/'experiments/results/credit'/f'{a.tag}.json';assert not out.exists(),'Use a fresh result tag'
    result=contracts(a) if a.contract else run(a)
    result.update(tag=a.tag,battle='B1/R1 shared reciprocal-learning enabler',
                  source_sha256={str(Path(__file__).resolve().relative_to(ROOT)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
                  max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  hardware=dict(host=socket.gethostname(),torch=torch.__version__,threads=torch.get_num_threads()))
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result.get('checks',result.get('evaluation')),indent=2),flush=True)


if __name__=='__main__':main()
