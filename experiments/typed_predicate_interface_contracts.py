"""Small stdlib semantic contracts; interface evidence only, no trained model."""
import json
import math
from sleeping_machines.typed_predicate_interface import Field,Predicate,TypedPredicateBank

def contracts():
    numeric=Field('temperature','numeric');category=Field('material','categorical',('steel','wood'))
    threshold=Predicate('hot',numeric,threshold=20.,scale=10.)
    membership=Predicate('metal',category,members=frozenset({'steel'}))
    boolean=Predicate('enabled',Field('enabled','boolean'))
    bank=TypedPredicateBank([threshold,membership,boolean])
    results,work=bank.evaluate_selected({'temperature':30.,'material':'steel'},['hot','metal'])
    assert [r.outcome for r in results]==['true','true'] and work['selected_comparisons']==2 and work['available_predicates']==3
    for result in results:assert min(result.race_delays(),key=result.race_delays().get)==result.outcome
    assert threshold.compare({'temperature':20.}).outcome=='false'
    near=Predicate('positive',numeric).compare({'temperature':1e-20})
    assert min(near.race_delays(),key=near.race_delays().get)==near.outcome
    changed_units=Predicate('hot',numeric,threshold=68.,scale=18.)
    assert math.isclose(threshold.compare({'temperature':30.}).margin,changed_units.compare({'temperature':86.}).margin)
    assert threshold.compare({'temperature':0.}).outcome=='false'
    assert threshold.compare({'temperature':None}).outcome=='missing'
    assert threshold.compare({'temperature':float('nan')}).outcome=='missing'
    assert membership.compare({'material':'glass'}).outcome=='unknown'
    relabel=Predicate('metal',Field('material','categorical',('z','a')),members=frozenset({'z'}))
    assert membership.compare({'material':'steel'}).outcome==relabel.compare({'material':'z'}).outcome
    for pred,row in [(threshold,{'temperature':True}),(boolean,{'enabled':1})]:
        try:pred.compare(row)
        except TypeError:pass
        else:raise AssertionError('Type mismatch accepted')
    a,_=bank.evaluate_selected({'temperature':30.,'material':'steel','unused':-999},['hot','metal'])
    assert a==results
    return dict(unit_equivalence=True,category_relabel_equivalence=True,missing_distinct_from_zero=True,
        unknown_category_explicit=True,type_mismatch_rejected=True,unselected_field_not_read=True,
        race_comparison_consistent=True,threshold_tie_policy_explicit=True,
        scope='Fixed schema/comparison interface only; no learned predicate selection, threshold learning, persistent event integration, counterfactual credit or benchmark quality claim.')

if __name__=='__main__':print(json.dumps(contracts(),indent=2))
