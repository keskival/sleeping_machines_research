"""Sparse supported residual comparison of issued, frozen route policies."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Comparison:
    # Tuples prevent later policy/prediction mutation from changing an issued audit.
    entries: tuple  # (route, p0-p, prediction)
    predicted_gain: float
    l1_distance: float
    version: tuple  # caller declares actor/learner/critic/history/horizon identity

    def proposal(self):
        return tuple((route, abs(weight)/self.l1_distance) for route, weight, _ in self.entries)

    def corrected_gain(self, route, observed_cost, version):
        if tuple(version) != self.version:
            raise ValueError('Audit target version/history/horizon mismatch')
        if not math.isfinite(observed_cost):
            raise ValueError('Nonfinite observed cost')
        for address, weight, prediction in self.entries:
            if address == route:
                return self.predicted_gain + math.copysign(self.l1_distance, weight)*(observed_cost-prediction)
        raise ValueError('Route outside issued comparison support')


def issue(prior, policy, predictions, version):
    """Input policies are sparse probability maps; no scan of unrepresented routes."""
    for distribution in [prior, policy]:
        if not distribution or any(not math.isfinite(p) or p < 0 for p in distribution.values()):
            raise ValueError('Invalid sparse policy')
        if not math.isclose(sum(distribution.values()), 1., rel_tol=0., abs_tol=1e-12):
            raise ValueError('Policy mass must be one')
    entries = []
    for route in dict.fromkeys([*prior, *policy]):
        weight = prior.get(route, 0.)-policy.get(route, 0.)
        if weight == 0: continue
        if route not in predictions or not math.isfinite(predictions[route]):
            raise ValueError('Every changed route needs a finite issued prediction')
        entries.append((route, weight, float(predictions[route])))
    return Comparison(tuple(entries), sum(w*q for _, w, q in entries),
                      sum(abs(w) for _, w, _ in entries), tuple(version))
