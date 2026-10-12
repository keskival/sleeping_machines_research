"""B10 finite-difference contract for the actual future-utility boundary vector."""
import argparse,hashlib,json,resource,time
from pathlib import Path
import torch
from torch.func import functional_call
from market_prospective_bias_credit import model_data,control,packet,Window,adam_proposal,ROOT

def objective(base,t,m,phi,audit):
    import copy
    model=copy.deepcopy(base);opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=0)
    # Nonzero warm Adam moments avoid the initialization's sharp denominator.
    _,_,_,warm,_,_=packet(model,t,m,None,0,8,0);torch.stack(warm).mean().backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
    produced,leaf,end,losses,_,_=packet(model,t,m,None,8,8,1);params=tuple(model.parameters())
    local=torch.autograd.grad(torch.stack(losses).mean(),params,retain_graph=True,allow_unused=True)
    target=torch.autograd.grad(torch.stack(losses[1:]).sum()/len(losses),control.tensors(leaf),retain_graph=True)
    calibration,prospective=phi.chunk(2)
    pred=[x.reshape_as(y) for x,y in zip(calibration.split([x.numel() for x in control.tensors(produced)]),control.tensors(produced))]
    credit=[h+(g.detach()-h)/.25 for h,g in zip(pred,target)] if audit else pred
    boundary=torch.autograd.grad(control.tensors(produced),params,grad_outputs=credit,create_graph=True,allow_unused=True)
    g=[(torch.zeros_like(p) if l is None else l.detach())+(torch.zeros_like(p) if b is None else b) for p,l,b in zip(params,local,boundary)]
    proposed=adam_proposal(model,opt,g)
    return functional_call(Window(model),proposed,(t,m,control.detach_state(end),16,8,2))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args();torch.set_num_threads(1);begin=time.monotonic();out=ROOT/f'experiments/results/credit/{a.tag}.json';assert not out.exists();base,t,m=model_data(187,80,32);width=sum(x.numel() for x in control.tensors(control.market_consume(base,t[:1],m[:1])[1]));torch.manual_seed(190);phi=(torch.randn(2*width,dtype=torch.float64)*.001).requires_grad_();direction=torch.randn_like(phi);direction/=direction.norm();rows=[]
    g=torch.randn(width,dtype=torch.float64);c,b=phi.chunk(2);p=.25;mean=(1-p)*(c+b)+p*(c+(g-c)/p+b);assert torch.allclose(mean,g+b,atol=1e-12,rtol=0)
    for audit in [False,True]:
        loss=objective(base,t,m,phi,audit);g=torch.autograd.grad(loss,phi)[0];analytic=float(g@direction);eps=1e-6
        checks=[]
        for eps in [1e-3,1e-4,1e-5,1e-6]:
            plus=objective(base,t,m,(phi.detach()+eps*direction).requires_grad_(),audit);minus=objective(base,t,m,(phi.detach()-eps*direction).requires_grad_(),audit);finite=float((plus-minus).detach()/(2*eps));checks.append(dict(eps=eps,finite_difference=finite,error=abs(analytic-finite)))
        assert min(c["error"] for c in checks[:2])<1e-5*max(1.,abs(analytic)),(analytic,checks)
        rows.append(dict(audit=audit,analytic=analytic,checks=checks))
        print(rows[-1],flush=True)
    sources=[Path(__file__),ROOT/'experiments/credit/market_prospective_bias_credit.py'];out.write_text(json.dumps(dict(status='completed',battle='B10',tag=a.tag,rows=rows,whole_wall_s=time.monotonic()-begin,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},scope='Actual native producer VJP, clipped Adam with fixed prior moments, next-window loss; both residual audit branches. Boundary-vector derivative, not all-history derivative.'),indent=2)+'\n');print(rows)
if __name__=='__main__':main()
