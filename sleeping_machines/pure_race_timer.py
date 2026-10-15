"""Forward-native pure timer with winner-averaged exact timing credit.

Only use when the winning clock's identity does not select a different value,
state write or future operation. Native value-routing races need their existing
coupled credit; this is not a replacement for TemporalRoute.
"""
import torch


class PureRaceTimer(torch.autograd.Function):
    @staticmethod
    def forward(ctx,scores):
        rates=scores.to(torch.float64).exp()
        time,winner=(torch.empty_like(rates).exponential_()/rates).min(0)
        ctx.save_for_backward(rates,time)
        ctx.input_dtype=scores.dtype
        return .001+.010*time/(1+time),winner

    @staticmethod
    def backward(ctx,error_delay,error_winner):
        rates,time=ctx.saved_tensors
        if error_delay is None:return torch.zeros_like(rates,dtype=ctx.input_dtype)
        common=-error_delay*.010*time/(1+time).square()
        return (common*rates/rates.sum()).to(ctx.input_dtype)
