"""Factored forward eligibility for the actual native TemporalUnit write law.

No temporal autograd tape. Exact own-write-parameter derivatives conditional on
fixed parameters, input messages, routing decisions and event timestamps.
Cross-receiver/input-producer/clock-selection credit is separate, not claimed.
Only a proposed receiver's traces are computed; callers commit only the winner.
"""
from dataclasses import dataclass
import math
import torch
from torch.nn import functional as F


@dataclass
class ReceiverEligibility:
    value: torch.Tensor
    input_real: torch.Tensor
    input_imag: torch.Tensor
    control_weight: torch.Tensor
    control_bias: torch.Tensor
    raw_rate: torch.Tensor
    frequency: torch.Tensor
    timestamp: float | None = None

    @classmethod
    def empty(cls, unit):
        p=unit.payload; w=unit.input.weight
        if p%2 or w.dtype not in (torch.float32,torch.float64):
            raise ValueError('even native payload and real float32/64 parameters required')
        dtype=torch.complex128 if w.dtype==torch.float64 else torch.complex64
        z=lambda *shape:torch.zeros(*shape,dtype=dtype,device=w.device)
        return cls(z(p//2),z(p//2,p),z(p//2,p),z(p//2,2,p),z(p//2,2),z(p//2),z(p//2))

    @torch.no_grad()
    def propose(self, unit, incoming, timestamp):
        p=unit.payload;x=incoming.detach();stamp=float(timestamp)
        if x.shape!=(p,) or not bool(torch.isfinite(x).all()) or not math.isfinite(stamp):
            raise ValueError('finite native message and timestamp required')
        if self.timestamp is not None and stamp<self.timestamp:
            raise ValueError('receiver event times must be nondecreasing')
        elapsed=0. if self.timestamp is None else stamp-self.timestamp
        normalized=F.layer_norm(x,(p,));controls=unit.control(normalized)
        forget=F.softplus(controls[0])/math.log(2);sig=controls[1].sigmoid();write=2*sig
        rate=F.softplus(unit.raw_rate)+1e-6
        # Match the native unit's float64 clock angles even with float32
        # payloads. A long dormant interval must not lose its phase precision.
        magnitude=torch.exp(-elapsed*rate*forget)
        angle=elapsed*unit.frequency.double()
        rotation=torch.complex(magnitude*angle.cos().to(x.dtype),magnitude*angle.sin().to(x.dtype))
        transported=rotation*self.value
        projected=unit.input(x).reshape(-1,2);proposal=torch.complex(projected[:,0],projected[:,1])
        dc_forget=transported*(-elapsed*rate*controls[0].sigmoid()/math.log(2))
        dc_write=proposal*(2*sig*(1-sig))
        dc=torch.stack((dc_forget,dc_write),-1)
        return ReceiverEligibility(
            transported+write*proposal,
            rotation[:,None]*self.input_real+write*x[None],
            rotation[:,None]*self.input_imag+1j*write*x[None],
            rotation[:,None,None]*self.control_weight+dc[:,:,None]*normalized[None,None],
            rotation[:,None]*self.control_bias+dc,
            rotation*self.raw_rate-transported*elapsed*forget*unit.raw_rate.sigmoid(),
            rotation*self.frequency+1j*elapsed*transported,
            stamp)

    def memory(self):
        return torch.view_as_real(self.value).flatten().detach()

    @torch.no_grad()
    def credit(self, cotangent):
        """A later local read's state cotangent credits earlier writes directly."""
        if cotangent.shape!=(2*len(self.value),) or not bool(torch.isfinite(cotangent).all()):
            raise ValueError('finite native memory cotangent required')
        g=cotangent.detach().reshape(-1,2);g=torch.complex(g[:,0],g[:,1]).conj()
        project=lambda trace:(g.reshape((len(g),)+(1,)*(trace.ndim-1))*trace).real
        wr=project(self.input_real);wi=project(self.input_imag)
        return {'input.weight':torch.stack((wr,wi),1).flatten(0,1),
                'control.weight':project(self.control_weight).sum(0),
                'control.bias':project(self.control_bias).sum(0),
                'raw_rate':project(self.raw_rate),'frequency':project(self.frequency)}

    def storage(self):
        fields=('value','input_real','input_imag','control_weight','control_bias','raw_rate','frequency')
        return dict(tensor_bytes=sum(getattr(self,k).numel()*getattr(self,k).element_size() for k in fields),
                    complex_scalars=sum(getattr(self,k).numel() for k in fields),
                    temporal_graph_entries=0,
                    scope='One occupied receiver: numerical state and factored own-write eligibility; parameters, routing credit, metadata and readout work separate.')

    @staticmethod
    @torch.no_grad()
    def sum_credit(records):
        """Batch matured messages for one receiver; same conditional trace law."""
        if not records:raise ValueError('at least one local credit message required')
        traces,cotangents=zip(*records)
        gs=torch.stack(cotangents).reshape(len(records),-1,2)
        g=torch.complex(gs[...,0],gs[...,1]).conj()
        def project(name):
            values=torch.stack([getattr(t,name) for t in traces])
            shaped=g.reshape(g.shape+(1,)*(values.ndim-2))
            return (shaped*values).real.sum(0)
        return {'input.weight':torch.stack((project('input_real'),project('input_imag')),1).flatten(0,1),
                'control.weight':project('control_weight').sum(0),
                'control.bias':project('control_bias').sum(0),
                'raw_rate':project('raw_rate'),'frequency':project('frequency')}
