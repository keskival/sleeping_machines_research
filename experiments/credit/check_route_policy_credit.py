"""Enumerate sampled updates against the true expected route-cost gradient."""
import argparse, hashlib, json, resource, time
from pathlib import Path
import torch
from route_policy_credit import route_credit_objective
ROOT=Path(__file__).resolve().parents[2]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True)
    a=ap.parse_args();torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
    start=time.monotonic();out=ROOT/f'experiments/results/credit/{a.tag}.json'
    assert not out.exists();checks=[]
    for routes in [3, 17, 65]:
        for exploration in [0., .1, 1.]:
            torch.manual_seed(8300+routes)
            logits=torch.randn(routes,requires_grad=True);costs=torch.randn(routes)
            p=(1-exploration)*logits.softmax(0)+exploration/routes
            truth=torch.autograd.grad((p*costs).sum(),logits)[0]
            for predictor in ['zero','arbitrary','perfect']:
                h={'zero':torch.zeros_like(costs),'arbitrary':torch.randn(routes),
                   'perfect':costs.clone()}[predictor].requires_grad_()
                expected=torch.zeros_like(logits);variance=0.;grads=[]
                for action in range(routes):
                    objective=route_credit_objective(logits,h,action,costs[action],exploration)
                    gradient,critic_gradient=torch.autograd.grad(objective,(logits,h),allow_unused=True)
                    assert critic_gradient is None
                    expected+=p.detach()[action]*gradient;grads.append(gradient)
                for action,g in enumerate(grads):variance+=float(p.detach()[action]*(g-truth).square().sum())
                error=float((expected-truth).abs().max());assert error<1e-12
                if predictor=='perfect':assert variance<1e-24
                checks.append(dict(routes=routes,exploration=exploration,predictor=predictor,
                                   gradient_error=error,conditional_variance=variance))
    paths=['experiments/credit/route_policy_credit.py','experiments/credit/check_route_policy_credit.py']
    result=dict(status='completed',battle='R1',tag=a.tag,gate=True,checks=checks,
                whole_wall_s=time.monotonic()-start,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},
                scope='Exhaustive3/17/65-route issued-logit gradients, zero/arbitrary/perfect predictors and0/.1/1 uniform-mixture exploration. Predictor detached; exact conditional expected-cost gradient and perfect-model action-noise removal. No task fit or native-parameter credit claim.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
