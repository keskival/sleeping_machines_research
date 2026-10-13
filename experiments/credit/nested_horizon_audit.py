"""R1 streaming paired-loss ledger for a selected finite-horizon receipt.

This records caller-produced outcomes; it does not simulate learner worlds.
"""
from dataclasses import dataclass
import math
from finite_horizon_receipt import HorizonReceipt


@dataclass(frozen=True)
class PairedOutcome:
    step: int
    reference_observation: str
    intervention_observation: str
    reference_loss: float
    intervention_loss: float
    objective_id: str
    world_id: str
    learner_version: str


class NestedHorizonAudit:
    """One continuous paired stream; two boundary scalars, no trajectory buffer.

    Weights are frozen for the full terminal objective before outcomes arrive.
    Outcome identities enforce common exogenous observations, suitable for
    corpus prediction. Action-dependent observation worlds need another API.
    """
    def __init__(self, receipt: HorizonReceipt, level: int, weights):
        if type(level) is not int or not 0 <= level < len(receipt.horizons):
            raise ValueError('Selected level is outside the issued ladder')
        self.receipt = receipt
        self.level = level
        self.weights = tuple(weights)
        if len(self.weights) != receipt.horizons[-1] or any(
                not math.isfinite(w) or w < 0 for w in self.weights):
            raise ValueError('Freeze one finite nonnegative weight per terminal outcome')
        self.lower_horizon = receipt.horizons[level-1] if level else 0
        self.upper_horizon = receipt.horizons[level]
        self.count = 0
        self.total = 0.
        self.compensation = 0.
        self.lower_value = 0. if level == 0 else None

    def append(self, outcome: PairedOutcome):
        if self.count >= self.upper_horizon:
            raise ValueError('Selected audit already matured')
        if type(outcome.step) is not int or outcome.step != self.count + 1:
            raise ValueError('Paired outcomes must be contiguous and ordered')
        if not outcome.reference_observation or (
                outcome.reference_observation != outcome.intervention_observation):
            raise ValueError('Both losses must use the same exogenous observation')
        if (outcome.objective_id, outcome.world_id, outcome.learner_version) != (
                self.receipt.objective_id, self.receipt.world_id, self.receipt.learner_version):
            raise ValueError('Outcome does not belong to this issued cut')
        if not all(math.isfinite(x) for x in (outcome.reference_loss, outcome.intervention_loss)):
            raise ValueError('Paired losses must be finite')
        increment = self.weights[self.count] * (outcome.reference_loss - outcome.intervention_loss)
        if not math.isfinite(increment):
            raise ValueError('Weighted loss difference overflow')
        # Compensated sum; validate before mutating so rejected input is atomic.
        adjusted = increment - self.compensation
        total = self.total + adjusted
        if not math.isfinite(total):
            raise ValueError('Cumulative advantage overflow')
        compensation = (total - self.total) - adjusted
        self.total, self.compensation = total, compensation
        self.count += 1
        if self.count == self.lower_horizon:
            self.lower_value = self.total

    def target(self, *, completion_probability=1.):
        if self.count != self.upper_horizon:
            raise ValueError('Incomplete audit cannot supply a completed correction')
        return self.receipt.corrected_target(
            self.level, self.lower_value, self.total,
            objective_id=self.receipt.objective_id, world_id=self.receipt.world_id,
            learner_version=self.receipt.learner_version,
            completion_probability=completion_probability)
