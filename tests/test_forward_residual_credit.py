"""B1/R1 proof checks: finite exact expectations; no optimizer or model fit."""
import itertools
import math
import pytest
from experiments.credit.forward_residual_credit import (allocation, add, enumerate_pairs, full_credit,
                                                        norm2, paired_credit, scale, score_vectors)


@pytest.mark.parametrize('count', [2, 3, 8])
def test_arbitrary_predictor_preserves_shared_parameter_mean(count):
    p = [(i+1)/sum(range(1,count+1)) for i in range(count)]
    f = [math.sin(i)+i/3 for i in range(count)]
    b = [(-1)**i * (i+2) for i in range(count)]
    h = score_vectors(p, [[1, math.cos(i), i*i/9] for i in range(count)])
    q = [[0 if i==w else .8*p[i]/(1-p[w])+.2/(count-1) for i in range(count)] for w in range(count)]
    initial = enumerate_pairs(p,f,b,h,q)
    assert initial['maximum_mean_error'] < 1e-12
    improved = [.75*x+.25*y+7 for x,y in zip(f,b)]
    result = enumerate_pairs(p,f,improved,h,q)
    assert result['maximum_mean_error'] < 1e-12
    assert result['variance_trace'] == pytest.approx(initial['variance_trace']/16,abs=1e-12)
    perfect = enumerate_pairs(p,f,[x+17 for x in f],h,q)
    assert perfect['variance_trace'] < 1e-25
    if count==2:
        for w in range(count):
            assert paired_credit(p,f,b,h,w,1-w,1) == pytest.approx(initial['target'],abs=1e-12)


def test_independent_audits_have_the_declared_variance():
    p,f,b = [.2,.3,.5],[1.,-2.,4.],[.7,-1.,2.]
    h=score_vectors(p,[[1.,2.],[2.,-.4],[-1.,.7]])
    target=full_credit(p,f,h); base=full_credit(p,b,h)
    mean=[0.,0.]; variance=0.; formula=0.
    for w,pw in enumerate(p):
        alternatives=[i for i in range(3) if i!=w]; inclusion=[.25,.7]
        corrections=[scale(h[i],p[i]*((f[i]-f[w])-(b[i]-b[w]))) for i in alternatives]
        formula+=pw*sum((1/chance-1)*norm2(a) for a,chance in zip(corrections,inclusion))
        for bits in itertools.product((0,1),repeat=2):
            chance=pw; value=base[:]
            for bit,inc,a in zip(bits,inclusion,corrections):
                chance*=inc if bit else 1-inc
                if bit:value=add(value,scale(a,1/inc))
            mean=add(mean,scale(value,chance)); variance+=chance*norm2(add(value,scale(target,-1)))
    assert mean==pytest.approx(target,abs=1e-12)
    assert variance==pytest.approx(formula,abs=1e-12)


def test_cost_allocation_satisfies_budget_and_interior_optimality():
    magnitude,cost,budget=[.2,.3,.5],[1.,4.,9.],4.
    probability=allocation(magnitude,cost,budget)
    assert sum(p*c for p,c in zip(probability,cost))==pytest.approx(budget,abs=1e-12)
    multipliers=[a*a/(c*p*p) for a,c,p in zip(magnitude,cost,probability)]
    assert max(multipliers)-min(multipliers)<1e-12
    with pytest.raises(ValueError):allocation(magnitude,cost,.01)


def test_equal_messages_can_have_nonzero_write_credit():
    p=[.4,.6]; h=score_vectors(p,[[1.],[0.]])
    loss=[math.log1p(math.exp(-1)),math.log(2)]
    predictions=[1.,1.]   # identical outgoing messages conceal the private write
    target=full_credit(p,loss,h)
    assert target[0]<-.08
    assert paired_credit(p,loss,predictions,h,0,1,1)==pytest.approx(target,abs=1e-12)


def test_forward_packet_rotation_and_decay_matches_later_baseline():
    p = [.2, .3, .5]
    h = score_vectors(p, [[1., 2.], [2., -.4], [-1., .7]])
    features = [[1., 0.], [0., 2.], [-.5, .7]]
    angle, decay = .8, .6
    a = [[decay*math.cos(angle), -decay*math.sin(angle)],
         [decay*math.sin(angle), decay*math.cos(angle)]]
    outcome = [.4, -.8]
    packet = [[sum(pi*hi[j]*fi[k] for pi,hi,fi in zip(p,h,features))
               for k in range(2)] for j in range(2)]
    transported = [[sum(row[k]*a[l][k] for k in range(2))
                    for l in range(2)] for row in packet]
    predicted = [sum(row[k]*outcome[k] for k in range(2)) for row in transported]
    utilities = [sum(outcome[l]*sum(a[l][k]*fi[k] for k in range(2))
                     for l in range(2)) for fi in features]
    assert predicted == pytest.approx(full_credit(p, utilities, h), abs=1e-12)


def test_sample_dependent_predictor_can_bias_the_mean():
    p, f = [.2, .3, .5], [1., -2., 4.]
    h = score_vectors(p, [[1.], [2.], [-1.]])
    mean = 0.
    for w in range(3):
        for i in range(3):
            if i == w:
                continue
            # Illegally refit only the currently audited alternative.
            b = [0., 0., 0.]
            b[i] = f[i]
            mean += p[w]*.5*paired_credit(p,f,b,h,w,i,.5)[0]
    assert abs(mean-full_credit(p,f,h)[0]) > .1
