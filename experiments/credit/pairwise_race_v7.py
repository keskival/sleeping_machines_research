#!/usr/bin/env python3
"""B1/R1 shared route-credit enabler: exact pairwise odds train the credit model;
independence MH corrects proposed causes for forward training. Persistent per-item
causes versus a fresh proposal start. Fixed-model stationary correctness is NOT
finite-step exactness or convergence during neural learning. No benchmark TEST.
"""
import argparse
import hashlib
import json
import resource
import socket
import time
from pathlib import Path
import torch
import torch.nn.functional as F
from hindsight_race_v6 import Net, Credit, make_data, evaluate, credit_gap

ROOT=Path(__file__).resolve().parents[2]


def pair_loss(logq_i,logq_j,loga_i,loga_j):
    """Forward target detached; q learns exact posterior conditional on the pair."""
    return F.binary_cross_entropy_with_logits(logq_j-logq_i,torch.sigmoid((loga_j-loga_i).detach()))


def log_acceptance(loga_i,loga_j,logproposal_i,logproposal_j):
    return ((loga_j-loga_i)+(logproposal_i-logproposal_j)).clamp_max(0)


def run(a):
    torch.set_num_threads(1);start=time.time()
    out=ROOT/'experiments/results/credit'/f'{a.tag}.json'
    if out.exists(): raise ValueError('Result exists; use a fresh tag')
    data=make_data(a,torch.Generator().manual_seed(a.data_seed))
    xtr,ytr,ktr,xdev,ydev,kdev,teacher_ll=data
    digest=hashlib.sha256()
    for t in data[:-1]:digest.update(t.contiguous().numpy().tobytes())
    torch.manual_seed(a.seed);g=torch.Generator().manual_seed(a.seed)
    m=Net(a.d,a.K,a.C,a.depth,a.K2);q=Credit(a.d,a.C,m.n_exp,h=a.credit_hidden)
    opt=torch.optim.Adam(m.parameters(),lr=a.lr);qopt=torch.optim.Adam(q.parameters(),lr=a.lr)
    causes=torch.randint(m.n_exp,(a.n_train,),generator=g)
    router=a.d*a.K+(a.d*a.K*a.K2 if a.depth==2 else 0)
    expert=a.C*a.d;h=a.credit_hidden;credit=a.d*h+h*h+h*m.n_exp
    macs=0;visits=0;history=[]
    for ep in range(a.epochs):
        permutation=torch.randperm(a.n_train,generator=g)
        accepted=0;distinct=0;moved=0;pair_bce=0.;draws=0
        for offset in range(0,a.n_train,a.batch):
            ix=permutation[offset:offset+a.batch];x=xtr[ix];y=ytr[ix];B=len(ix)
            lpri=m.route_logprior(x);lq=q(x,y)
            with torch.no_grad():
                proposal=(1-a.explore)*lq.exp()+a.explore/m.n_exp
                logproposal=proposal.log()
                old=causes[ix] if a.arm=='pairwise_persistent' else torch.multinomial(proposal,1,generator=g).squeeze(-1)
                candidate=torch.multinomial(proposal,1,generator=g).squeeze(-1)
                loga_i=lpri.gather(-1,old[:,None]).squeeze(-1)+m.expert_logp(x,y,old)
                loga_j=lpri.gather(-1,candidate[:,None]).squeeze(-1)+m.expert_logp(x,y,candidate)
                laccept=log_acceptance(loga_i,loga_j,logproposal.gather(-1,old[:,None]).squeeze(-1),logproposal.gather(-1,candidate[:,None]).squeeze(-1))
                accept=torch.rand(B,generator=g).clamp_min(1e-30).log()<laccept
                chosen=torch.where(accept,candidate,old)
                causes[ix]=chosen
                accepted+=accept.sum().item();distinct+=(candidate!=old).sum().item();moved+=(chosen!=old).sum().item()
            qloss=pair_loss(lq.gather(-1,old[:,None]).squeeze(-1),lq.gather(-1,candidate[:,None]).squeeze(-1),loga_i,loga_j)
            qopt.zero_grad();qloss.backward();qopt.step()
            visits+=sum(p.numel() for p in q.parameters() if p.grad is not None)
            # Re-evaluate only the chosen expert with gradients; score-function
            # complete-data gradient is exact in expectation at posterior stationarity.
            loss=-(lpri.gather(-1,chosen[:,None]).squeeze(-1)+m.expert_logp(x,y,chosen)).mean()
            opt.zero_grad();loss.backward();opt.step()
            visits+=sum(p.numel() for p in m.parameters() if p.grad is not None)
            macs+=B*(3*router+5*expert+3*credit)
            pair_bce+=B*qloss.item();draws+=B
        ll,purity=evaluate(m,xdev,ydev,kdev,m.n_exp);kl,tv=credit_gap(m,q,xdev,ydev)
        row=dict(epoch=ep,dev_marginal_ll=ll,route_purity=purity,credit_kl=kl,credit_tv=tv,
                 mh_acceptance=accepted/draws,distinct_pair_fraction=distinct/draws,move_fraction=moved/draws,
                 pair_bce=pair_bce/draws,macs_per_example=macs/((ep+1)*a.n_train))
        history.append(row);print(json.dumps(row),flush=True)
    files=[Path(__file__),Path(__file__).with_name('hindsight_race_v6.py')]
    res=dict(status='completed',tag=a.tag,battle='B1/R1 shared route-credit enabler',args=vars(a),
             data_sha256=digest.hexdigest(),history=history,teacher_dev_marginal_ll=teacher_ll,
             final_dev_marginal_ll=history[-1]['dev_marginal_ll'],final_credit_kl=history[-1]['credit_kl'],final_credit_tv=history[-1]['credit_tv'],
             final_route_purity=history[-1]['route_purity'],macs_per_example=history[-1]['macs_per_example'],whole_fit_linear_macs=macs,
             expert_evals_per_example=3,optimizer_parameter_visits=visits,optimizer_parameter_visits_per_example=visits/(a.epochs*a.n_train),
             evaluation_linear_macs=a.epochs*a.n_test*(2*(router+m.n_exp*expert)+credit),
             work_convention='leading-linear MAC estimate; 3 router + 5 expert + 3 credit; nonlinear, bias, sampling and optimizer FLOPs excluded',
             selection='fixed final epoch on synthetic DEV; no benchmark test',
             mechanism_scope='hard-route credit diagnostic; no integrated persistent temporal memory',
             posterior_scope='fixed-model MH invariant posterior; finite-step/moving-target bias remains',
             cause_storage_bytes=causes.numel()*causes.element_size(),wall_s=time.time()-start,
             max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
             hardware=dict(host=socket.gethostname(),threads=torch.get_num_threads(),torch=torch.__version__),
             source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    with out.open('x') as f:json.dump(res,f,indent=2);f.write('\n')
    return res


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True)
    ap.add_argument('--arm',choices=['pairwise_persistent','pairwise_reset'],default='pairwise_persistent')
    for name,default in [('depth',2),('K',8),('K2',4),('d',16),('C',10),('n-train',20000),('n-test',5000),('epochs',10),('batch',64),('credit-hidden',8),('seed',0),('data-seed',1234)]:
        ap.add_argument('--'+name,type=int,default=default)
    ap.add_argument('--lr',type=float,default=.003);ap.add_argument('--explore',type=float,default=.1)
    a=ap.parse_args()
    if min(a.n_train,a.n_test,a.epochs,a.batch,a.credit_hidden,a.K,a.K2,a.d,a.C)<1 or a.depth not in (1,2) or not 0<a.explore<1:ap.error('invalid positive dimensions, depth or exploration floor')
    run(a)
