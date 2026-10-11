"""Compose exact temporal teaching packets with one shared elapsed-time input.

Each layer uses the same elapsed intervals. Its clock cotangent contributes to
the shared producer exactly once. Packet parameter graphs may remain connected;
these methods expose first-order input/time transport, not parameter transport.
"""
import torch

from native_temporal_teaching_packet import NativeTemporalTeachingPacket
from native_training_temporal_packet import NativeTrainingTemporalPacket


class NativeDeepTemporalPacket:
    @classmethod
    def capture(cls, layers, content, elapsed, *, context, version, connected=False):
        if not layers:
            raise ValueError('at least one layer required')
        packet_type = NativeTrainingTemporalPacket if connected else NativeTemporalTeachingPacket
        self = cls()
        self.packets = []
        h = content
        for layer in layers:
            packet = packet_type.capture(layer, h, elapsed, context=context, version=version)
            self.packets.append(packet)
            h = packet.forward_output if connected else packet.output
        self.output = h
        return self

    @torch.no_grad()
    def apply_with_time(self, directions, time_directions, *, context, version):
        for packet in self.packets:
            directions = packet.apply_with_time(directions, time_directions,
                                                context=context, version=version)
        return directions

    @torch.no_grad()
    def transpose_with_time(self, cotangents, *, context, version):
        elapsed = torch.zeros_like(cotangents[..., 0])
        for packet in reversed(self.packets):
            cotangents, local_elapsed = packet.transpose_with_time(
                cotangents, context=context, version=version)
            elapsed.add_(local_elapsed)
        return cotangents, elapsed

    def cached_floats(self):
        return sum(packet.cached_floats() for packet in self.packets)
