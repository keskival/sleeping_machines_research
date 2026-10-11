"""B1 contract: conditional mean, risk gradient, and constrained audit allocation."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from predictable_audit_allocation_v2 import allocate,ResidualRiskLearner
ROOT=Path(__file__).resolve().parents[2]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args()
 pi=np.array([.1,.2,.3,.4]);v=np.array([.01,.25,4.,16.]);cost=np.array([1.,2.,3.,4.]);budget=1.2
 p=allocate(pi,v,cost,budget,floor=.02);work=float(pi@(cost*p));m2=float(pi@(v/p));uniform=budget/float(pi@cost);baseline=float(pi@(v/uniform))
 assert abs(work-budget)<1e-10 and np.all(p>=.02) and m2<=baseline+1e-10
 # Interior KKT condition and boundary direction.
 ratios=v/(cost*p*p);interior=(p>.02000001)&(p<.99999999);k=float(np.mean(ratios[interior]));assert np.max(abs(ratios[interior]-k))<1e-9
 assert np.all(ratios[p<=.02000001]<=k+1e-9) and np.all(ratios[p>=.99999999]>=k-1e-9)
 # Enumerated Bernoulli audit: issue once, then average consumed estimators.
 residual=np.array([-.2,.5,1.,-2.]);h=np.array([.3,.2,-.1,.5]);scores=np.array([1.,-1.,2.,-1.]);base=float(pi@(h*scores));exact=base+float(pi@(residual*scores));estimate=base+sum(float(pi[i])*(p[i]*(residual[i]*scores[i]/p[i])) for i in range(4));assert abs(exact-estimate)<1e-14
 learner=ResidualRiskLearner(4);gradient_error=0
 for i in range(4):
  packet=learner.issue(np.eye(4)[i],p[i]);observed=learner.gradient(packet,audited=True,residual_squared=v[i]);unobserved=learner.gradient(packet,audited=False)
  full=learner.gradient(learner.issue(np.eye(4)[i],1),audited=True,residual_squared=v[i]);gradient_error=max(gradient_error,float(np.max(abs(p[i]*observed+(1-p[i])*unobserved-full))))
 assert gradient_error<1e-14
 # Consume an issued delayed packet after another update: issued-gradient identity remains.
 packet=learner.issue([1,0,0,0],.5);old=learner.gradient(packet,audited=True,residual_squared=2.)
 learner.consume(learner.issue([1,0,0,0],1),audited=True,residual_squared=5.,step=.1)
 assert np.array_equal(old,learner.gradient(packet,audited=True,residual_squared=2.))
 zero=allocate(pi,np.zeros(4),cost,budget);assert np.all(zero==.02)
 for bad in [0,10]:
  try:allocate(pi,v,cost,bad)
  except ValueError:pass
  else:raise AssertionError('infeasible budget admitted')
 extreme=allocate(np.full(4,.25),[1e-300,1e300,0,1],[1e300,1e-300,1,1],1e298)
 assert np.isfinite(extreme).all() and np.all(extreme>=.02-1e-15)
 assert abs(float(np.full(4,.25)@(np.array([1e300,1e-300,1,1])*extreme))/1e298-1)<1e-10
 out=dict(status='completed',battle='B1/R1 audit economy contract',tag=a.tag,propensities=p.tolist(),expected_incremental_work=work,uniform_probability=uniform,second_moment=m2,uniform_second_moment=baseline,second_moment_reduction=1-m2/baseline,conditional_mean_error=abs(exact-estimate),risk_gradient_mean_error=gradient_error,source_sha256={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in ['experiments/credit/check_predictable_audit_allocation_v2.py','experiments/credit/predictable_audit_allocation_v2.py']},scope='Exact finite synthetic contract with oracle risk inputs. No native learned-risk calibration, model quality, wall/FLOP/depth scaling or TEST result. Work units are arbitrary supplied costs; computing inputs/controller is additional.')
 with (ROOT/'experiments/results/credit'/f'{a.tag}.json').open('x') as f:f.write(json.dumps(out,indent=2)+'\n')
 print(json.dumps(out),flush=True)
if __name__=='__main__':main()
