"""B1/R1 finite mean/covariance contracts for sparse model-term credit."""
import argparse,hashlib,itertools,json,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args()
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64);start=time.monotonic()
    out=ROOT/f'experiments/results/credit/{a.tag}.json';assert not out.exists()
    logits=torch.tensor([.2,-.1,.4],requires_grad=True);pi=.9*logits.softmax(0)+.1/3;prob=pi.detach();rho=torch.tensor([.2,.5,.3])
    score=torch.stack([torch.autograd.grad(pi[k].log(),logits,retain_graph=True)[0] for k in range(3)]).detach()
    d=torch.tensor([.03,-.02,.01]);exact=torch.autograd.grad((pi*d).sum(),logits,retain_graph=True)[0];rows=[]
    for name,h in [('imperfect',torch.tensor([.01,.04,-.01])),('perfect',d)]:
        z=prob[:,None]/rho[:,None]*h[:,None]*score
        residual=(d-h)[:,None]*score
        gh=(rho[:,None]*z).sum(0);gr=(prob[:,None]*residual).sum(0)
        vz=float((rho[:,None]*(z-gh).square()).sum());vr=float((prob[:,None]*(residual-gr).square()).sum())
        for j in [1,2,4]:
            mean=torch.zeros_like(exact);moment=0.
            for cs in itertools.product(range(3),repeat=j):
                pc=float(rho[list(cs)].prod());model=z[list(cs)].mean(0)
                for action in range(3):
                    g=model+residual[action];weight=pc*float(prob[action]);mean+=weight*g;moment+=weight*float((g-exact).square().sum())
            error=float((mean-exact).abs().max());cov_error=abs(moment-(vz/j+vr));assert max(error,cov_error)<1e-14
            rows.append(dict(predictor=name,queries=j,mean_error=error,covariance_error=cov_error,covariance_trace=moment))
        # Coupled C=A cancels to ordinary policy credit, independently of predictor quality.
        coupled=h[:,None]*score+residual;plain=d[:,None]*score
        assert float((coupled-plain).abs().max())<1e-14
    c0,c,vbase,vmodel=2.,.05,.003,.12
    optimum=(c0*vmodel/(c*vbase))**.5
    derivative=c*vbase-c0*vmodel/optimum**2;assert abs(derivative)<1e-14
    value=lambda j:(c0+c*j)*(vbase+vmodel/j)
    assert value(optimum)<=min(value(optimum*.5),value(optimum*2))
    result=dict(status='completed',battle='B1/R1',tag=a.tag,gate=True,contracts=rows,coupled_query_cancellation=True,budget_stationary_point=optimum,budget_derivative_error=abs(derivative),whole_wall_s=time.monotonic()-start,source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['experiments/credit/check_sparse_model_term.py','experiments/theory/SPARSE_MODEL_TERM_CREDIT_ECONOMICS.md']},scope='Exhaustive finite three-action independent proposal/actual-action enumeration,1/2/4 queries and exact policy-autograd mean. Sparse-model covariance decomposition, perfect/imperfect predictor, coupled cancellation and continuous iid work-variance optimum. No native model fitting, hardware/depth or measured economical query budget claim.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
