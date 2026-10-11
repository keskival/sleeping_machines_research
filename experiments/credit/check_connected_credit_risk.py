"""R1 enumerate residual critic-risk loss/parameter-gradient identity."""
import argparse,hashlib,json,time
import torch
from check_token_forward_credit import ROOT
from connected_credit_risk import corrected_credit_risk

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args();start=time.monotonic();out=ROOT/f'experiments/results/credit/{a.tag}.json';assert not out.exists()
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64);rows=[]
    for depth in [2,4,8]:
        torch.manual_seed(99140+depth);layers=[torch.nn.Linear(5,8),torch.nn.Tanh()]
        for _ in range(depth-1):layers.extend([torch.nn.Linear(8,8),torch.nn.Tanh()])
        layers.append(torch.nn.Linear(8,1));critic=torch.nn.Sequential(*layers);x=torch.randn(9,5);costs=torch.randn(9);t=(torch.randn(9)*3).requires_grad_();q=torch.randn(9).softmax(0).requires_grad_();w=torch.rand(9)
        h=critic(x).flatten();true=(w*(h-costs).square()).sum();expected=sum(q[k].detach()*corrected_credit_risk(h,t,k,costs[k],q,w) for k in range(9))
        params=list(critic.parameters());gtrue=torch.autograd.grad(true,params,retain_graph=True);gest=torch.autograd.grad(expected,params,retain_graph=True)
        label_grads=torch.autograd.grad(expected,[t,q],allow_unused=True)
        assert all(v is None for v in label_grads)
        errors=[float((true-expected).abs().detach())]+[float((u-v).abs().max()) for u,v in zip(gtrue,gest)];assert max(errors)<1e-10
        rows.append(dict(depth=depth,max_error=max(errors)))
    result=dict(status='completed',battle='R1',tag=a.tag,gate=True,checks=rows,wall_s=time.monotonic()-start,source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['experiments/credit/connected_credit_risk.py','experiments/credit/check_connected_credit_risk.py']})
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
