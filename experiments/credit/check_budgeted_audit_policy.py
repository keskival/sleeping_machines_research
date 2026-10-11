"""B1 finite-budget implicit policy credit, conservation and probe-gradient mean."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import time

import torch

from budgeted_audit_policy import budgeted_probabilities, forward_audit_policy_loss

ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--tag',required=True)
    args=parser.parse_args();torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
    torch.manual_seed(1050);start=time.monotonic();rows=[]
    for count in [4,32,128]:
        scores=torch.randn(count,requires_grad=True);risk=torch.rand(count)+.1;budget=count*.125
        p=budgeted_probabilities(scores,budget)
        loss=forward_audit_policy_loss(scores,risk,budget)
        gradient,=torch.autograd.grad(loss,scores)
        finite=[];eps=1e-5
        for index in range(count):
            direction=torch.zeros(count);direction[index]=eps
            plus=forward_audit_policy_loss(scores.detach()+direction,risk,budget)
            minus=forward_audit_policy_loss(scores.detach()-direction,risk,budget)
            finite.append(float((plus-minus)/(2*eps)))
        finite_error=float((gradient-torch.tensor(finite)).abs().max())
        shifted=budgeted_probabilities(scores.detach()+7,budget)
        shift_error=float((p.detach()-shifted).abs().max())
        budget_error=abs(float(p.sum().detach())-budget)
        # These positive observations have the declared mean; policy credit is linear in labels.
        observed=[]
        for label in [risk*.5,risk*1.5]:
            g,=torch.autograd.grad(forward_audit_policy_loss(scores,label,budget),scores)
            observed.append(g)
        mean_error=float(((observed[0]+observed[1])/2-gradient).abs().max())
        assert max(finite_error,shift_error,budget_error,mean_error,abs(float(gradient.sum())))<1e-8
        assert float(p.min())>.01 and float(p.max())<1
        rows.append(dict(contexts=count,budget=budget,finite_difference_error=finite_error,
                         shared_shift_error=shift_error,budget_error=budget_error,
                         forward_label_gradient_mean_error=mean_error,gradient_sum=float(gradient.sum())))
    sources=['experiments/credit/check_budgeted_audit_policy.py','experiments/credit/budgeted_audit_policy.py']
    result=dict(status='completed',tag=args.tag,battle='B1 directly trained audit policy',rows=rows,
                source_sha256={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sources},
                wall_s=time.monotonic()-start,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                scope='Analytic implicit first-order budget-policy credit vs numerical derivatives, positive support/conservation/common-shift invariance and frozen-label gradient mean. No native fit/task utility/higher-order or full-learning efficiency claim.')
    with (ROOT/'experiments/results/credit'/f'{args.tag}.json').open('x') as output:json.dump(result,output,indent=2)
    print(json.dumps(result))


if __name__=='__main__':main()
