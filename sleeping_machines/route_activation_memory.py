"""Bounded causal distribution summaries for native route-credit inputs.

Numerical observation memory, not a credit predictor. Summaries use normalized
exponential mixture weights; effective count is not a calibration guarantee.
"""
from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class ActivationSnapshot:
    mean: np.ndarray
    variance: np.ndarray
    projected_covariance: np.ndarray
    observations: int
    effective_count: float
    last_time: float
    age: float
    actor_version: str
    version_changes: int
    operator_stale: bool


class RouteActivationMemory:
    def __init__(self, width, projection, max_routes=1024):
        projection = np.asarray(projection, dtype=np.float64)
        if width <= 0 or max_routes <= 0 or projection.ndim != 2:
            raise ValueError('Positive dimensions and a matrix projection required')
        if projection.shape[1] != width or not np.isfinite(projection).all():
            raise ValueError('Projection must be finite and match activation width')
        self.width, self.max_routes = width, max_routes
        self.projection = projection.copy()
        self.records = {}
        self.updates = self.reads = 0

    def observe(self, route, activation, time, actor_version, retention=.99):
        """Observe a causal activation; caller chooses retention explicitly.

        Equal-time events remain separate observations. Invalid input cannot
        mutate a record. Memory never evicts evidence silently at capacity.
        """
        h = np.asarray(activation, dtype=np.float64)
        if h.shape != (self.width,) or not np.isfinite(h).all():
            raise ValueError('Finite activation vector required')
        if not math.isfinite(time) or not 0 <= retention < 1:
            raise ValueError('Finite time and retention in [0,1) required')
        if not isinstance(actor_version, str) or not actor_version:
            raise ValueError('Explicit actor version required')
        old = self.records.get(route)
        if old is not None and time < old['time']:
            raise ValueError('Out-of-order observation would corrupt causal history')
        if old is None and len(self.records) >= self.max_routes:
            raise MemoryError('Route budget exhausted; explicit retention policy required')
        if old is None:
            k = self.projection.shape[0]
            record = dict(mean=h.copy(), variance=np.zeros(self.width),
                          covariance=np.zeros((k, k)), weight_square=1., count=1,
                          time=float(time), version=actor_version, changes=0)
        else:
            delta = h - old['mean']
            projected = self.projection @ delta
            a = float(retention)
            record = dict(mean=a*old['mean']+(1-a)*h,
                          variance=a*old['variance']+a*(1-a)*delta**2,
                          covariance=a*old['covariance']+a*(1-a)*np.outer(projected, projected),
                          weight_square=a*a*old['weight_square']+(1-a)**2,
                          count=old['count']+1, time=float(time), version=actor_version,
                          changes=old['changes']+int(actor_version != old['version']))
        self.records[route] = record
        self.updates += 1

    def snapshot(self, route, time, actor_version):
        """Copy issued features so later observations cannot rewrite them."""
        if not math.isfinite(time) or not isinstance(actor_version, str) or not actor_version:
            raise ValueError('Finite query time and explicit actor version required')
        r = self.records.get(route)
        if r is None:
            return None
        if time < r['time']:
            raise ValueError('Cannot issue features using future observations')
        self.reads += 1
        return ActivationSnapshot(r['mean'].copy(), r['variance'].copy(),
                                  r['covariance'].copy(), r['count'], 1/r['weight_square'],
                                  r['time'], float(time-r['time']), r['version'],
                                  r['changes'], actor_version != r['version'])

    def work(self):
        return dict(routes=len(self.records), updates=self.updates, reads=self.reads,
                    numeric_bytes=self.projection.nbytes+sum(
                        r['mean'].nbytes+r['variance'].nbytes+r['covariance'].nbytes
                        for r in self.records.values()),
                    scope='Numeric arrays only; excludes Python objects and issued copies. '
                          'Update O(k*d+k^2+d), record O(k^2+d); no route discovery performed.')
