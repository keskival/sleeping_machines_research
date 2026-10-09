"""B1/R1 route-credit enabler contracts: pairwise credit and stationary MH."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from pairwise_race_v7 import Net,pair_loss,log_acceptance
from pairwise_math import audit,transition

ROOT=Path(__file__).resolve().parents[2]


def checks():
    torch.set_num_threads(1)
    rho=torch.tensor([.6,.3,.1],dtype=torch.float64);proposal=torch.tensor([.05,.15,.8],dtype=torch.float64)
    la=rho.log();lq=proposal.log()
    acceptance=log_acceptance(la[:,None],la[None,:],lq[:,None],lq[None,:]).exp()
    kernel=acceptance*proposal[None,:]
    kernel.fill_diagonal_(0);kernel.diagonal().copy_(1-kernel.sum(-1))
    expected=torch.tensor(transition(rho.tolist(),proposal.tolist()),dtype=torch.float64)
    kernel_error=(kernel-expected).abs().max().item();assert kernel_error<1e-14
    logits=rho.log().requires_grad_();scores=rho.log().requires_grad_()
    pairs=torch.tensor([[0,1],[0,2],[1,2]])
    loss=pair_loss(logits[pairs[:,0]],logits[pairs[:,1]],scores[pairs[:,0]],scores[pairs[:,1]])
    grad,teacher_grad=torch.autograd.grad(loss,(logits,scores),allow_unused=True)
    assert teacher_grad is None and grad.abs().max()<1e-14
    wrong=proposal.log().requires_grad_()
    wrong_loss=pair_loss(wrong[pairs[:,0]],wrong[pairs[:,1]],scores[pairs[:,0]],scores[pairs[:,1]])
    assert torch.autograd.grad(wrong_loss,wrong)[0].abs().max()>.1
    errors={}
    for depth in (1,2):
        torch.manual_seed(51)
        m=Net(4,3,3,depth,2).double();x=torch.randn(5,4,dtype=torch.float64);y=torch.tensor([0,1,2,1,0])
        lpri=m.route_logprior(x);le=m.expert_logp(x,y)
        exact=torch.autograd.grad(-torch.logsumexp(lpri+le,-1).mean(),tuple(m.parameters()))
        lpri=m.route_logprior(x);le=m.expert_logp(x,y);posterior=(lpri+le).softmax(-1).detach()
        expected_grad=torch.autograd.grad(-(posterior*(lpri+le)).sum(-1).mean(),tuple(m.parameters()))
        err=max((a-b).abs().max().item() for a,b in zip(exact,expected_grad))
        assert err<1e-12;errors[f'depth{depth}_stationary_all_parameter_gradient_error']=err
    return dict(kernel_error=kernel_error,pair_posterior_gradient_max=grad.abs().max().item(),
                target_detached=True,**errors,finite_state=audit())


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args()
    result=dict(status='completed',battle='B1/R1 shared route-credit enabler',tag=a.tag,checks=checks())
    sources=[Path(__file__),Path(__file__).with_name('pairwise_race_v7.py'),Path(__file__).with_name('pairwise_math.py'),Path(__file__).with_name('hindsight_race_v6.py')]
    result['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    with (ROOT/'experiments/results/credit'/f'{a.tag}.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result['checks'],indent=2))
