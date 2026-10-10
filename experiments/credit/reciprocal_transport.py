"""Local reciprocal transport prototype; explicit fixed-context/version packets.

Dense matrix storage/update is charged. This is not a context-generalizing
critic or an exact future cotangent model. All teaching arithmetic is local.
"""
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class DirectionalPacket:
    context: str
    version: int
    direction: torch.Tensor
    response: torch.Tensor

    @classmethod
    def capture(cls, context, version, direction, response):
        return cls(context, version, direction.detach().clone(), response.detach().clone())


class ReciprocalTransport:
    def __init__(self, matrix, *, context, version):
        assert matrix.ndim == 2 and torch.isfinite(matrix).all()
        self.matrix = matrix.detach().clone()
        self.context = context
        self.version = version
        self.updates = 0

    def _check(self, context, version):
        if context != self.context or version != self.version:
            raise ValueError('Transport context/version mismatch; no silent stale-credit reuse')

    def predict(self, cotangent, *, context, version):
        self._check(context, version)
        if cotangent.shape != (self.matrix.shape[1],):
            raise ValueError('Output cotangent shape mismatch')
        return self.matrix @ cotangent

    @torch.no_grad()
    def teach(self, packet, cotangent, *, step):
        self._check(packet.context, packet.version)
        if packet.direction.shape != (self.matrix.shape[0],) or packet.response.shape != (self.matrix.shape[1],):
            raise ValueError('Directional packet shape mismatch')
        if cotangent.shape != packet.response.shape:
            raise ValueError('Teaching cotangent shape mismatch')
        if step <= 0:
            raise ValueError('Step must be positive')
        if not all(bool(torch.isfinite(x).all()) for x in (packet.direction, packet.response, cotangent)):
            raise ValueError('Nonfinite teaching packet')
        delta = packet.direction @ self.matrix @ cotangent - cotangent @ packet.response
        update = step * delta * torch.outer(packet.direction, cotangent)
        if not bool(torch.isfinite(update).all()) or not bool(torch.isfinite(self.matrix-update).all()):
            raise ValueError('Nonfinite reciprocal update')
        self.matrix.sub_(update); self.updates += 1
        return float(delta)
