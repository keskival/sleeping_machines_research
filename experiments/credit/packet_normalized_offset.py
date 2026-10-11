"""Clockless additive calibration updates normalized to one issued causal packet."""
from dataclasses import dataclass
import math

from policy_difference_audit import issue


@dataclass(frozen=True)
class NormalizedReceipt:
    packet: object
    raw_predictions: tuple
    target_version: tuple
    identifier: tuple
    issued_offset: float
    learning_weight: float


class PacketNormalizedOffset:
    def __init__(self, initial=0., rate=.125):
        if not math.isfinite(initial) or not 0 < rate <= 1:
            raise ValueError('Invalid calibration state/rate')
        self.value, self.rate = float(initial), float(rate)
        self.version = 0
        self.pending = {}

    def issue_batch(self, prior, policy, predictions, target_version, identifier, count):
        if not isinstance(count, int) or count < 1:
            raise ValueError('Positive feedback sample count required')
        keys = [(identifier, j) for j in range(count)]
        if any(key in self.pending for key in keys):
            raise ValueError('Batch identifier already pending')
        offset, version = self.value, self.version
        receipts = []
        for key in keys:
            packet = issue(prior, policy, {a: p+offset for a, p in predictions.items()},
                           (tuple(target_version), version, key))
            receipt = NormalizedReceipt(packet, tuple((a, float(predictions[a])) for a, _, _ in packet.entries),
                                        tuple(target_version), key, offset, self.rate/count)
            receipts.append(receipt)
            if packet.entries:
                self.pending[key] = receipt
        return receipts

    def consume(self, receipt, route, observed, target_version):
        if tuple(target_version) != receipt.target_version:
            raise ValueError('Wrong actor/critic/history/horizon target')
        if self.pending.get(receipt.identifier) is not receipt:
            raise ValueError('Unknown or consumed feedback')
        gain = receipt.packet.corrected_gain(route, observed, receipt.packet.version)
        raw = dict(receipt.raw_predictions)[route]
        # An issued gradient carries its own calibration coordinate and mass.
        # Additive updates commute without a barrier between feedback arrivals.
        self.value += receipt.learning_weight*(observed-raw-receipt.issued_offset)
        self.version += 1
        del self.pending[receipt.identifier]
        return gain
