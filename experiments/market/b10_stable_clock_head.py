"""Algebraic log-scale evaluation of the original recording-cell clock law.

The floor c*exp(-mu) can overflow before multiplication, poisoning a zero
cotangent as 0*inf. Keeping sigma in log space preserves the intended finite
clock law without clipping mu or weakening the resolution floor.
"""
import math
import torch
from torch.nn import functional as F
from race_tpp_v5 import RaceTPP,LOG2PI

class LogScaleRaceHead(RaceTPP):
    def clocks(self,h,slots):
        B,L,_=h.shape;raw=self.clock(torch.cat((h,slots.flatten(-2)),-1));ne,nl=self.n_exp,self.n_ln
        log_rate=raw[...,:ne]-math.log(self.scale);mu=raw[...,ne:ne+nl]+math.log(self.scale)
        spread=raw[...,ne+nl:ne+2*nl]
        # For x<-36 log(softplus(x)) rounds to x in binary64. Clamp only the
        # inactive evaluation branch to avoid log(0), never the model parameter.
        log_spread=torch.where(spread < -36,spread,F.softplus(spread.clamp_min(-36)).log())
        log_floor=(math.log(self.floor_cell)-mu).clamp_min(math.log(.005)) if self.floor_cell>0 else torch.full_like(mu,math.log(.005))
        log_sigma=torch.logaddexp(log_spread,log_floor)
        fire=raw[...,ne+2*nl:ne+3*nl];log_w=F.log_softmax(raw[...,ne+3*nl:],-1);log_w=torch.cat((log_w[...,:ne],F.logsigmoid(fire)),-1)
        logits=self.mark_ctx(h).view(B,L,self.M,self.K)+torch.einsum('blkv,mv->blmk',slots,self.mark_slot)
        return log_rate,mu,log_sigma,log_w,F.log_softmax(logits,-1)
    @staticmethod
    def clock_terms(tau,log_rate,mu,log_sigma,log_w):
        tau=tau.unsqueeze(-1);ne=log_rate.shape[-1];w=log_w[...,:ne].exp()
        ls_exp=-log_rate.exp()*tau*w;lh_exp=(log_rate+log_w[...,:ne]).expand_as(ls_exp)
        log_pi=log_w[...,ne:];log_1mpi=torch.log1p(-log_pi.exp().clamp_max(1-1e-12));pos=tau>0;lt=torch.log(tau.clamp_min(1e-30))
        z=(lt-mu)*torch.exp(-log_sigma);log_s0=torch.special.log_ndtr(-z)
        log_f0=-lt-log_sigma-.5*LOG2PI-.5*z**2;log_s=torch.logaddexp(log_1mpi,log_pi+log_s0)
        lh_ln=torch.where(pos,log_pi+log_f0-log_s,torch.full_like(z,-math.inf));ls_ln=torch.where(pos,log_s,torch.zeros_like(z))
        return torch.cat((lh_exp,lh_ln),-1),torch.cat((ls_exp,ls_ln),-1)

def install_log_scale_head(model):
    if type(model.m) is LogScaleRaceHead:return model
    if type(model.m) is not RaceTPP:raise ValueError('Original native head required')
    # Parameter ownership, names, optimizer state and encoded information stay
    # unchanged. Bound alternative encoders' encode method is preserved.
    model.m.__class__=LogScaleRaceHead
    return model
