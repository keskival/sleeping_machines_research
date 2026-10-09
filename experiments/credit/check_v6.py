"""Numerical contracts for the B1/R1 shared route-credit enabler; no fitting."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hindsight_race_v6 as H
from estimator_audit import audit


def checks():
    torch.set_num_threads(1)
    a = SimpleNamespace(d=4, K=3, C=3, depth=2, K2=2, n_train=12, n_test=8, data_seed=1234)
    def data(initial_seed):
        torch.manual_seed(initial_seed)
        return H.make_data(a, torch.Generator().manual_seed(a.data_seed))
    first, second = data(1), data(999)
    assert all(torch.equal(x,y) if isinstance(x,torch.Tensor) else x==y for x,y in zip(first,second))
    a.data_seed += 1
    assert not torch.equal(first[1],data(1)[1]) or not torch.equal(first[0],data(1)[0])
    errors = {}
    for depth in (1,2):
        torch.manual_seed(5)
        m = H.Net(4,3,3,depth,2).double()
        x, y = torch.randn(5,4,dtype=torch.float64), torch.tensor([0,1,2,1,0])
        exact_loss = -torch.logsumexp(m.route_logprior(x)+m.expert_logp(x,y),-1).mean()
        exact = torch.autograd.grad(exact_loss, tuple(m.parameters()))
        prior, expert = m.route_logprior(x), m.expert_logp(x,y)
        rho = (prior+expert).softmax(-1).detach()
        # Every cause once, uniform proposal: check ALL router and expert gradients.
        surrogate = -(rho*(prior+expert)).sum(-1).mean()
        got = torch.autograd.grad(surrogate,tuple(m.parameters()))
        err = max((a-b).abs().max().item() for a,b in zip(exact,got))
        assert err < 1e-12
        errors[f'depth_{depth}_all_parameter_max_error'] = err
        q = H.Credit(4,3,m.n_exp,h=8).double()
        lq = q(x,y)
        torch.manual_seed(7)
        k = torch.multinomial(lq.detach().exp(),2,replacement=True)
        le = torch.stack([m.expert_logp(x,y,k[:,i]) for i in range(2)],-1)
        lp = m.route_logprior(x).gather(-1,k)
        w = ((lp+le).detach()-lq.detach().gather(-1,k)).softmax(-1)
        qloss = -(w*lq.gather(-1,k)).sum(-1).mean()
        assert all(g is None for g in torch.autograd.grad(qloss,tuple(m.parameters()),allow_unused=True,retain_graph=True))
        assert all(torch.isfinite(g).all() for g in torch.autograd.grad(qloss,tuple(q.parameters())))
        forward_loss = -(w*(lp+le)).sum(-1).mean()
        assert all(g is None for g in torch.autograd.grad(forward_loss,tuple(q.parameters()),allow_unused=True,retain_graph=True))
        assert all(torch.isfinite(g).all() for g in torch.autograd.grad(forward_loss,tuple(m.parameters())))
    return dict(teacher_independent_of_learner_rng=True, detached_update_paths=True, **errors, math_audit=audit())


if __name__ == '__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--tag',required=True); a=ap.parse_args()
    res=dict(status='completed',tag=a.tag,battle='B1/R1 shared route-credit enabler',checks=checks())
    root=Path(__file__).resolve().parents[2]
    sources=[Path(__file__),Path(H.__file__),Path(__file__).with_name('estimator_audit.py')]
    res['source_sha256']={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    out=root/'experiments/results/credit'/f'{a.tag}.json'
    with out.open('x') as f: json.dump(res,f,indent=2); f.write('\n')
    print(json.dumps(res['checks'],indent=2))
