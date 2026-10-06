"""Stdlib semantic/math witnesses; no fit, tensor runtime or inference job."""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sleeping_machines.typed_predicate_interface import Field, Predicate

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists(): raise FileExistsError(output)
    count = 0
    numeric = Field('temperature', 'numeric')
    original = Predicate('above', numeric, threshold=2.5, scale=.5)
    expected = original.compare({'temperature':3.5})
    for a,b in itertools.product((.125,1.,8.,32.),(-64.,0.,128.)):
        changed = Predicate('above', numeric, threshold=a*2.5+b, scale=a*.5)
        actual = changed.compare({'temperature':a*3.5+b})
        assert actual == expected and actual.race_delays() == expected.race_delays()
        count += 1
    outcomes = []
    raw = []
    embeddings = []
    for codes in itertools.permutations((1,2,3)):
        mapping = dict(zip(('red','green','blue'),codes))
        category = Field('color','categorical',categories=codes)
        predicate = Predicate('is_red',category,members=frozenset((mapping['red'],)))
        outcomes.append(predicate.compare({'color':mapping['red']}).race_delays())
        raw.append(.5*mapping['red']+7.)
        table = {mapping[name]:vector for name,vector in
                 [('red',(1.,0.)),('green',(0.,1.)),('blue',(-1.,-1.))]}
        embeddings.append(table[mapping['red']])
        count += 1
    assert all(row == outcomes[0] for row in outcomes)
    assert len(set(raw)) > 1
    assert all(row == embeddings[0] for row in embeddings)
    assert original.compare({'temperature':None}).outcome == 'missing'
    assert original.compare({'temperature':0.}).outcome == 'false'
    try: original.compare({'temperature':True})
    except TypeError: pass
    else: raise AssertionError('Boolean coercion into numeric field')
    count += 3
    # A monotone transform preserves a logical cut, not its affine margin.
    warped = Predicate('above',numeric,threshold=2.5**3,scale=.5)
    actual = warped.compare({'temperature':3.5**3})
    assert actual.outcome == expected.outcome and actual.margin != expected.margin
    count += 1
    # Without residual delivery both these observations yield the same bit.
    cut = Predicate('above_zero',numeric,threshold=0.,scale=1.)
    assert cut.compare({'temperature':-2.}).outcome == cut.compare({'temperature':-1.}).outcome
    count += 1
    derivative_errors = []
    for m in (-2.,-.3,.2,1.7):
        x,t,s = m,0.,1.
        def risk(threshold, scale=1., at=.3, af=1.2, B=.4):
            margin=(x-threshold)/scale
            rt,rf=math.exp(margin),math.exp(-margin)
            total=rt+rf;p=rt/total
            return p*at+(1-p)*af+B/total
        rt,rf=math.exp(m),math.exp(-m);total=rt+rf;p=rt/total
        gradient_m=2*p*(1-p)*(.3-1.2)-.4*(rt-rf)/(total*total)
        analytic=-gradient_m/s
        epsilon=1e-6
        measured=(risk(t+epsilon)-risk(t-epsilon))/(2*epsilon)
        error=abs(analytic-measured);derivative_errors.append(error)
        assert error < 1e-8
        for a,b in ((.125,4.),(8.,-16.)):
            xt,tt,st=a*x+b,a*t+b,a*s
            mt=(xt-tt)/st
            assert math.isclose(mt,m,abs_tol=1e-13)
            transformed=-gradient_m/st
            assert math.isclose(transformed,analytic/a,abs_tol=1e-13)
            count += 1
        # Equal branch risks leave the nonzero computation-time term.
        assert -.4*(rt-rf)/(total*total) != 0
        count += 2
    paths=[Path(__file__),ROOT/'sleeping_machines/typed_predicate_interface.py',
           ROOT/'experiments/theory/TYPED_SEMANTICS_AND_RACE_COMPOSITION_20261006.md',
           ROOT/'experiments/theory/typed_threshold_joint_credit_20261005.md']
    result=dict(status='completed',cases=count,affine_margin_clock_contract=True,
        nominal_schema_relabel_contract=True,raw_affine_encoding_obstruction=True,
        legal_embedding_remedy=True,missing_and_type_rejection=True,
        monotone_outcome_not_margin_witness=True,bit_compression_information_loss=True,
        joint_choice_and_time_derivative=True,threshold_gradient_unit_covariance=True,
        maximum_finite_difference_error=max(derivative_errors),
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        scope='Stdlib mathematical/interface contracts only. No trained threshold, sparse predicate discovery, integrated model fit or tree benchmark claim.')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
if __name__ == '__main__': main()
