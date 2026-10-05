"""Typed comparisons before event content mixing; no learned selector in this module."""
from dataclasses import dataclass
import math
from numbers import Real

@dataclass(frozen=True)
class Field:
    name: str
    kind: str
    categories: tuple = ()
    def __post_init__(self):
        if not self.name or self.kind not in ('numeric','categorical','boolean'):
            raise ValueError('Named numeric, categorical or boolean field required')
        if self.kind=='categorical':
            if not self.categories or len(set(self.categories))!=len(self.categories):
                raise ValueError('Distinct declared category labels required')
        elif self.categories:
            raise ValueError('Categories belong only to categorical fields')

@dataclass(frozen=True)
class Predicate:
    name: str
    field: Field
    threshold: float = 0.
    scale: float = 1.
    members: frozenset = frozenset()
    def __post_init__(self):
        if not self.name:raise ValueError('Predicate identity required')
        if self.field.kind=='numeric':
            if not math.isfinite(self.threshold) or not math.isfinite(self.scale) or self.scale<=0 or self.members:
                raise ValueError('Finite numeric threshold, positive scale and no category members required')
        elif self.field.kind=='categorical':
            if not self.members or not self.members.issubset(set(self.field.categories)):
                raise ValueError('Membership must use declared category labels')
        elif self.members:
            raise ValueError('Boolean comparison needs no membership set')
    def compare(self, row):
        value=row[self.field.name]
        if value is None:return Comparison(self.name,self.field.name,'missing',None)
        if self.field.kind=='numeric':
            if isinstance(value,bool) or not isinstance(value,Real):
                raise TypeError('Numeric field requires a numeric value, never a category ID or Boolean')
            if math.isnan(float(value)):return Comparison(self.name,self.field.name,'missing',None)
            if not math.isfinite(float(value)):raise ValueError('Infinite numeric value is not missing')
            margin=(float(value)-self.threshold)/self.scale
            if not math.isfinite(margin):raise ValueError('Nonfinite normalized comparison margin')
        elif self.field.kind=='categorical':
            if value not in self.field.categories:return Comparison(self.name,self.field.name,'unknown',None)
            margin=1. if value in self.members else -1.
        else:
            if type(value) is not bool:raise TypeError('Boolean field requires True/False')
            margin=1. if value else -1.
        return Comparison(self.name,self.field.name,'true' if margin>0 else 'false',margin)

@dataclass(frozen=True)
class Comparison:
    predicate: str
    field: str
    outcome: str
    margin: float | None
    def race_delays(self):
        """Two computational comparison clocks, not physical arrival timestamps.

        Missing/unknown take explicit separate channels. Deterministic tie -> false.
        This is a fixed comparison encoding, not stochastic route credit or a fit.
        """
        if self.margin is None:return {self.outcome:1.}
        m=max(-40.,min(40.,self.margin))
        false,true=math.exp(m),math.exp(-m)
        # Preserve strict logical ordering when exp rounds a tiny margin to a tie.
        if m>0 and true>=false:true=math.nextafter(false,0.)
        if m<0 and false>=true:false=math.nextafter(true,0.)
        return {'false':false,'true':true}

class TypedPredicateBank:
    def __init__(self,predicates):
        predicates=tuple(predicates)
        self.predicates={p.name:p for p in predicates}
        if len(self.predicates)!=len(predicates):raise ValueError('Distinct predicate IDs required')
        fields={}
        for predicate in predicates:
            prior=fields.setdefault(predicate.field.name,predicate.field)
            if prior!=predicate.field:raise ValueError('One consistent schema per field')
    def evaluate_selected(self,row,selected):
        selected=tuple(selected)
        if len(set(selected))!=len(selected):raise ValueError('Selected predicates must be distinct')
        results=tuple(self.predicates[name].compare(row) for name in selected)
        return results,dict(available_predicates=len(self.predicates),selected_comparisons=len(results),
            predicate_selection_work='Caller-owned; not implemented or included here')
