"""R1/B1 exhaustive numerical contract, no fitting or native quality claim."""
import argparse,hashlib,itertools,json,time
from pathlib import Path
from event_credit import probabilities,estimate
ROOT=Path(__file__).resolve().parents[2]
SOURCES=["experiments/credit/event_credit.py","experiments/credit/check_event_credit.py"]

def contract():
    actual=[[1.,-2.],[3.,4.],[-1.,2.]];pred=[[.8,-1.],[2.,4.],[0.,1.]]
    costs=[1.,4.,2.];risks=[sum((a-b)**2 for a,b in zip(x,y)) for x,y in zip(actual,pred)]
    rows=[]
    for p in ([3./7]*3,probabilities(risks,costs,3.)):
        mean=[0.,0.];variance=0.;target=[sum(v[k] for v in actual) for k in range(2)]
        for mask in itertools.product((0,1),repeat=3):
            mass=1.
            for bit,q in zip(mask,p):mass*=q if bit else 1-q
            value=estimate(pred,{i:actual[i] for i,b in enumerate(mask) if b},p)
            for k in range(2):mean[k]+=mass*value[k]
            variance+=mass*sum((a-b)**2 for a,b in zip(value,target))
        expected=sum((1/q-1)*v for q,v in zip(p,risks))
        error=max(abs(a-b) for a,b in zip(mean,target));assert error<1e-12 and abs(variance-expected)<1e-12
        rows.append(dict(probabilities=p,mean_error=error,variance=variance,variance_formula_error=abs(variance-expected),expected_audit_cost=sum(c*q for c,q in zip(costs,p))))
    assert rows[1]['expected_audit_cost']<=3.+1e-12
    assert rows[1]['variance']<=rows[0]['variance'] # equal expected audit cost
    for args in [([1.],[1.],.001),([-1.],[1.],1.),([1.],[0.],1.)]:
        try:probabilities(*args)
        except ValueError:pass
        else:raise AssertionError('invalid allocation accepted')
    try:estimate(pred,{},[0.,.5,.5])
    except ValueError:pass
    else:raise AssertionError('zero probability accepted')
    return dict(cases=rows,enumerated_masks=16,scope='Independent contribution audits, fixed version, full support. Discovery, native predictor cost and future utility are unmeasured.')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args();start=time.monotonic()
    result=contract();result.update(status='completed',tag=a.tag,battle='R1/B1',wall_s=time.monotonic()-start,source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCES})
    path=ROOT/'experiments/results/credit'/f'{a.tag}.json';assert not path.exists();path.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
