"""One forward packet teaches all output directions by a local rank-one projection."""
import torch
from reciprocal_transport import ReciprocalTransport


class ProjectedTransport(ReciprocalTransport):
    @torch.no_grad()
    def teach_packet(self, packet, *, relaxation=1.0):
        self._check(packet.context,packet.version)
        v,z=packet.direction,packet.response
        if v.shape!=(self.matrix.shape[0],) or z.shape!=(self.matrix.shape[1],):
            raise ValueError('Directional packet shape mismatch')
        if not 0<relaxation<2 or not all(bool(torch.isfinite(x).all()) for x in (v,z)):
            raise ValueError('Invalid relaxation or nonfinite packet')
        norm=v@v
        if not bool(norm>0):raise ValueError('Zero probe direction')
        residual=z-self.matrix.T@v
        update=(relaxation/norm)*torch.outer(v,residual)
        if not bool(torch.isfinite(self.matrix+update).all()):raise ValueError('Nonfinite projected update')
        self.matrix.add_(update);self.updates+=1
        return float(residual.square().sum())
