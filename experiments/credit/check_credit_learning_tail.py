"""Exact two-channel propagation and omitted discounted-credit tail contract."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args()
    out=ROOT/f'experiments/results/credit/{a.tag}.json';assert not out.exists();start=time.monotonic();checks=[]
    for rho,alpha in [(.5,.7),(.5,1.),(.5,.5)]:
        delta=.3;b=.2;u=delta;v=0.;error=0.;series=[]
        for j in range(200):
            exact=0. if j==0 else b*delta*(alpha**j-rho**j)/(alpha-rho) if alpha!=rho else b*delta*j*alpha**(j-1)
            error=max(error,abs(v-exact));series.append((u,v));u,v=rho*u,alpha*v+b*u
        assert error<1e-14
        gamma=.9;H=8;M=np.array([[rho,0.],[b,alpha]]);norm=float(np.linalg.norm(M,2));L=np.sqrt(2.)
        distance=float(np.linalg.norm(series[H]));tail=sum(gamma**j*sum(series[j]) for j in range(H,200))
        if gamma*norm<1:
            bound=gamma**H*L*distance/(1-gamma*norm);assert abs(tail)<=bound
        else:bound=None
        if alpha==1.:
            assert abs(series[-1][1]-b*delta/(1-rho))<1e-14 and abs(series[-1][0])<1e-50
        checks.append(dict(rho=rho,learning_gain=alpha,recurrence_error=error,addressed_tail=series[-1][0],learning_tail=series[-1][1],discounted_tail=tail,bound=bound,transition_spectral_norm=norm))
    paths=['experiments/credit/check_credit_learning_tail.py','experiments/theory/CAUSAL_CREDIT_HORIZON_ECONOMICS.md']
    result=dict(status='completed',battle='R1/B1',tag=a.tag,gate=True,checks=checks,whole_wall_s=time.monotonic()-start,source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},scope='Exact linear two-channel recurrence and spectral-norm discounted-tail bound. Neutral learning state retains influence after physical state decays. No native/Adam contraction or model-quality claim.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
