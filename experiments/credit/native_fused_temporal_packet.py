"""Exact joint content/time transport in one recurrence and readout sweep.

Primal capture and version discipline are unchanged. Joint methods share the
content/time linear recurrence, normalization and MLP rather than repeating
them. First-order local execution only; no learned credit or hard-route credit.
"""
import torch

from native_temporal_teaching_packet import NativeTemporalTeachingPacket
from native_training_linearization import NativeTrainingLinearization
from native_deep_temporal_packet import NativeDeepTemporalPacket


class NativeFusedTemporalPacket(NativeTemporalTeachingPacket):
    def _validate(self, block, *, context, version):
        if (context != self.context or version != self.version or
                self.parameter_versions != tuple(p._version for p in self.layer.parameters())):
            raise ValueError('stale native linearization')
        if self.layer.training or self.layer.drop.training:
            raise ValueError('training mode changed')
        if (block.ndim != 4 or tuple(block.shape[1:]) != tuple(self.shape) or
                not torch.isfinite(block).all()):
            raise ValueError('invalid joint transport block')

    @torch.no_grad()
    def apply_with_time(self, directions, time_directions, *, context, version):
        self._validate(directions, context=context, version=version)
        if (tuple(time_directions.shape) != tuple(directions.shape[:-1]) or
                not torch.isfinite(time_directions).all()):
            raise ValueError('invalid elapsed-time directions')
        v, w = directions.detach(), time_directions.detach()
        layer, n = self.layer, self.layer.n
        dw = v @ layer.write.weight.T
        dg = (v @ layer.gate.weight.T) * self.g * (1 - self.g)
        wr = dw[..., :n] * self.g + self.wr * dg
        wi = dw[..., n:] * self.g + self.wi * dg
        zr = v.new_zeros(v.shape[0], v.shape[1], n)
        zi = torch.zeros_like(zr)
        states = []
        for i in range(v.shape[2]):
            zr, zi = (self.cr[:, i] * zr - self.ci[:, i] * zi + wr[:, :, i] +
                      self.time_real[:, i] * w[:, :, i, None],
                      self.cr[:, i] * zi + self.ci[:, i] * zr + wi[:, :, i] +
                      self.time_imag[:, i] * w[:, :, i, None])
            states.append(torch.cat([zr, zi], -1))
        h = v + torch.stack(states, 2) @ layer.read.weight.T

        def norm_jvp(tangent, data):
            normalized, inv, weight = data
            return inv * (tangent - tangent.mean(-1, keepdim=True) -
                          normalized * (normalized * tangent).mean(-1, keepdim=True)) * weight

        h = norm_jvp(h, self.norm[0])
        mlp = (h @ layer.mlp[0].weight.T) * self.gelu_derivative
        return norm_jvp(h + mlp @ layer.mlp[2].weight.T, self.norm[1])

    @torch.no_grad()
    def transpose_with_time(self, cotangents, *, context, version):
        self._validate(cotangents, context=context, version=version)
        v = cotangents.detach()
        layer, n = self.layer, self.layer.n

        def norm_vjp(cotangent, data):
            normalized, inv, weight = data
            weighted = cotangent * weight
            return inv * (weighted - weighted.mean(-1, keepdim=True) -
                          normalized * (normalized * weighted).mean(-1, keepdim=True))

        pre2 = norm_vjp(v, self.norm[1])
        h = pre2 + ((pre2 @ layer.mlp[2].weight) * self.gelu_derivative) @ layer.mlp[0].weight
        direct = norm_vjp(h, self.norm[0])
        state = direct @ layer.read.weight
        sr = v.new_zeros(v.shape[0], v.shape[1], n)
        si = torch.zeros_like(sr)
        content = direct.clone()
        elapsed = v.new_zeros(v.shape[:-1])
        for i in range(v.shape[2] - 1, -1, -1):
            sr, si = sr + state[:, :, i, :n], si + state[:, :, i, n:]
            elapsed[:, :, i] = (sr * self.time_real[:, i] + si * self.time_imag[:, i]).sum(-1)
            g = self.g[:, i]
            writes = torch.cat([sr * g, si * g], -1)
            gates = (sr * self.wr[:, i] + si * self.wi[:, i]) * g * (1 - g)
            content[:, :, i] += writes @ layer.write.weight + gates @ layer.gate.weight
            sr, si = (self.cr[:, i] * sr + self.ci[:, i] * si,
                      -self.ci[:, i] * sr + self.cr[:, i] * si)
        return content, elapsed


class NativeFusedTrainingTemporalPacket(NativeFusedTemporalPacket, NativeTrainingLinearization):
    """Retain the original actor parameter graph during primal acquisition."""


class NativeDeepFusedTemporalPacket(NativeDeepTemporalPacket):
    @classmethod
    def capture(cls, layers, content, elapsed, *, context, version, connected=False):
        if not layers:
            raise ValueError('at least one layer required')
        packet_type = NativeFusedTrainingTemporalPacket if connected else NativeFusedTemporalPacket
        self = cls()
        self.packets = []
        h = content
        for layer in layers:
            packet = packet_type.capture(layer, h, elapsed, context=context, version=version)
            self.packets.append(packet)
            h = packet.forward_output if connected else packet.output
        self.output = h
        return self
