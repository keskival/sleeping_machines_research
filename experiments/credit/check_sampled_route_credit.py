"""B1/R1 finite universe enumeration, including routes outside chosen pool."""
import argparse,hashlib,json,math
from pathlib import Path
from sampled_route_credit import estimate
ROOT=Path(__file__).resolve().parents[2]
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args();out=ROOT/'experiments/results/credit'/f'{a.tag}.json';assert not out.exists()
    q=[.1,.2,.3,.25,.15];p=[.2,.5,.25,.8,.4];g=[[1.,0.],[-.5,1.],[2.,-.2],[.1,-.3],[-1.,.4]];h=[[.8,.1],[-.3,.8],[1.4,0.],[0.,-.1],[-.7,.2]];target=[sum(x[k] for x in g) for k in range(2)];mean=[0.,0.];variance=0.
    for i in range(5):
        for audited in (False,True):
            prob=q[i]*(p[i] if audited else 1-p[i]);value=estimate(q[i],p[i],h[i],g[i] if audited else None)
            for k in range(2):mean[k]+=prob*value[k]
            variance+=prob*sum((x-y)**2 for x,y in zip(value,target))
    expected=sum(sum(x*x for x in gi)/qi for gi,qi in zip(g,q))-sum(x*x for x in target)+sum((1/pi-1)*sum((x-y)**2 for x,y in zip(gi,hi))/qi for gi,hi,pi,qi in zip(g,h,p,q))
    error=max(abs(x-y) for x,y in zip(mean,target));verror=abs(variance-expected);assert error<1e-12 and verror<1e-12
    rejected=0
    for qi,pi in [(0.,.5),(.5,0.),(-.1,.5),(math.nan,.5)]:
        try:estimate(qi,pi,[1.])
        except ValueError:rejected+=1
    assert rejected==4
    sources=['experiments/credit/check_sampled_route_credit.py','experiments/credit/sampled_route_credit.py'];out.write_text(json.dumps(dict(status='completed',battle='B1/R1',tag=a.tag,metrics=dict(routes=5,chosen_pool=[0,1],exploration_outside_pool=[2,3,4],outcomes=10,mean_error=error,variance_error=verror,trace_variance=variance,support_rejections=rejected),source_sha256={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sources},scope='Exhaustive two-stage estimator arithmetic. No trained route generalization, native application, efficient discovery or hardware result.'),indent=2)+'\n')
if __name__=='__main__':main()
