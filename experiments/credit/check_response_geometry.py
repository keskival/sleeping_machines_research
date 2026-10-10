"""R1/B1 scalar response geometry contract, exhaustive endpoint cases plus samples."""
import argparse,hashlib,json,math,random,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

def terms(p,w,b,beta):
    den=[1+x*math.expm1(beta) for x in p];weights=[a/d for a,d in zip(w,den)];total=sum(weights);weights=[x/total for x in weights]
    probs=[x*math.exp(beta)/d for x,d in zip(p,den)];mean=sum(a*x for a,x in zip(weights,probs))
    variance=sum(a*(x-mean)**2 for a,x in zip(weights,probs));curvature=sum(a*x*(1-x) for a,x in zip(weights,probs))-variance
    return -b*beta-math.log(total),mean-b,curvature

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args();start=time.monotonic();rng=random.Random(181);cases=[([0.,1.],[.5,.5],0.,0.),([.5,.5],[.5,.5],1.,0.)]
    for _ in range(128):
        w=[rng.random() for _ in range(6)];total=sum(w);cases.append(([rng.random() for _ in w],[x/total for x in w],rng.random(),rng.uniform(-1,1)))
    error=0.;maximum=0.
    for p,w,b,beta in cases:
        value,grad,hess=terms(p,w,b,beta);h=1e-5;plus=terms(p,w,b,beta+h);minus=terms(p,w,b,beta-h)
        error=max(error,abs((plus[0]-minus[0])/(2*h)-grad),abs((plus[1]-minus[1])/(2*h)-hess));maximum=max(maximum,abs(hess));assert abs(grad)<=1+1e-12 and abs(hess)<=.25+1e-12 and abs(value)<=abs(beta)+1e-12
    assert error<1e-8;assert terms(*cases[0])[2]==-.25 and terms(*cases[1])[2]==.25
    source='experiments/credit/check_response_geometry.py';result=dict(status='completed',tag=a.tag,battle='R1/B1',cases=len(cases),maximum_finite_difference_error=error,maximum_absolute_curvature=maximum,wall_s=time.monotonic()-start,source_sha256={source:hashlib.sha256((ROOT/source).read_bytes()).hexdigest()},scope='Numerical derivative identities and sharp endpoint curvature cases; general inequalities are derived in STRUCTURED_RESPONSE_CREDIT.md. No learned uniform-error certificate or hardware result.')
    path=ROOT/'experiments/results/credit'/f'{a.tag}.json';assert not path.exists();path.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
