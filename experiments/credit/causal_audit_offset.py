"""A scalar audit-variance coordinate, separate from route decision contrasts."""
from dataclasses import dataclass
import math

from policy_difference_audit import issue as comparison


@dataclass(frozen=True)
class Receipt:
    packet: object
    raw_predictions: tuple
    target_version: tuple
    calibration_version: int
    identifier: object


class CausalOffset:
    def __init__(self, initial=0., rate=.125):
        if not math.isfinite(initial) or not 0 < rate <= 1:
            raise ValueError('Invalid calibration initial value/rate')
        self.value = float(initial)
        self.rate = float(rate)
        self.version = 0
        self.pending = {}

    def issue(self, prior, policy, predictions, target_version, identifier):
        if identifier in self.pending:
            raise ValueError('Audit identifier already pending')
        version = (tuple(target_version), self.version, identifier)
        shifted = {a: p+self.value for a, p in predictions.items()}
        packet = comparison(prior, policy, shifted, version)
        receipt = Receipt(packet, tuple((a, float(predictions[a])) for a, _, _ in packet.entries),
                          tuple(target_version), self.version, identifier)
        if packet.entries:
            self.pending[identifier] = receipt
        return receipt

    def consume(self, receipt, route, observed, target_version):
        if tuple(target_version) != receipt.target_version:
            raise ValueError('Wrong actor/critic/history/horizon target')
        if self.pending.get(receipt.identifier) is not receipt:
            raise ValueError('Unknown or already consumed audit receipt')
        gain = receipt.packet.corrected_gain(route, observed, receipt.packet.version)
        raw = dict(receipt.raw_predictions)[route]
        # Target is the residual relative to the issued RAW prediction.
        # Already-issued receipts retain their old calibration when feedback is delayed.
        self.value += self.rate*(observed-raw-self.value)
        self.version += 1
        del self.pending[receipt.identifier]
        return gain
