"""Orthogonal forward-probe block projection, explicit issued context/version.
Dense block acquisition/storage and orthogonality checks remain charged.
"""
import torch
from reciprocal_transport import ReciprocalTransport

class BulkProjectedTransport(ReciprocalTransport):
    @torch.no_grad()
    def teach_block(self,directions,responses,*,context,version,relaxation=1.):
        self._check(context,version)
        q,z=directions.detach(),responses.detach()
        if q.ndim!=2 or z.shape!=(q.shape[0],self.matrix.shape[1]) or q.shape[1]!=self.matrix.shape[0] or not 0<relaxation<2:raise ValueError('invalid probe block')
        if not torch.isfinite(q).all() or not torch.isfinite(z).all():raise ValueError('nonfinite block')
        gram=q@q.T
        if not torch.allclose(gram,torch.eye(q.shape[0],dtype=q.dtype,device=q.device),atol=1e-10,rtol=1e-10):raise ValueError('probe rows must be orthonormal')
        residual=z-q@self.matrix;update=relaxation*q.T@residual
        if not torch.isfinite(self.matrix+update).all():raise ValueError('nonfinite transport')
        self.matrix.add_(update);self.updates+=1
        return float(residual.square().sum())
