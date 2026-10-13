"""R1 finite telescoping receipt algebra; no rollout or training admission.

All baseline values are frozen at issue. The caller owns paired learner worlds,
version checks, objective weights, categorical sampling and complete work logs.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class HorizonReceipt:
    horizons: tuple[int, ...]
    predictions: tuple[float, ...]
    probabilities: tuple[float, ...]
    objective_id: str
    world_id: str
    learner_version: str

    def __post_init__(self):
        # Copy caller containers so delayed mutation cannot alter issued values.
        for name in ("horizons", "predictions", "probabilities"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        n = len(self.horizons)
        if not n or len(self.predictions) != n or len(self.probabilities) != n:
            raise ValueError("One prediction and probability per finite horizon")
        if any(type(h) is not int or h <= 0 for h in self.horizons):
            raise ValueError("Horizons must be positive integers")
        if any(a >= b for a, b in zip(self.horizons, self.horizons[1:])):
            raise ValueError("Horizons must strictly increase")
        if any(not math.isfinite(v) for v in self.predictions):
            raise ValueError("Issued predictions must be finite")
        if any(not math.isfinite(q) or q <= 0 for q in self.probabilities):
            raise ValueError("Every registered level needs positive support")
        if not math.isclose(math.fsum(self.probabilities), 1., rel_tol=0., abs_tol=1e-12):
            raise ValueError("Probabilities must sum to one")
        if not all((self.objective_id, self.world_id, self.learner_version)):
            raise ValueError("Objective, paired world and learner version are required")

    def corrected_target(self, level: int, lower_value: float, upper_value: float,
                         *, objective_id: str, world_id: str,
                         learner_version: str) -> float:
        """Stopped scalar target for the terminal head, using one nested audit.

        Values are cumulative weighted advantages, never per-horizon averages.
        IDs guard accidental receipt mixing; they do not validate replay fidelity.
        """
        if type(level) is not int or not 0 <= level < len(self.horizons):
            raise ValueError("Audit level outside registered ladder")
        if (objective_id, world_id, learner_version) != (
                self.objective_id, self.world_id, self.learner_version):
            raise ValueError("Audit does not match issued objective/world/version")
        if not all(math.isfinite(v) for v in (lower_value, upper_value)):
            raise ValueError("Audit labels must be finite")
        if level == 0 and lower_value != 0.:
            raise ValueError("The zero-horizon advantage must be zero")
        lower_prediction = self.predictions[level - 1] if level else 0.
        residual = (upper_value - lower_value) - (
            self.predictions[level] - lower_prediction)
        target = self.predictions[-1] + residual / self.probabilities[level]
        if not math.isfinite(target):
            raise ValueError("Importance correction overflow; preserve raw receipt")
        return target
