"""CPU content lookup with reverse-time, lane-ordered gradient aggregation.

Experimental execution backend. Full dense Adam work remains unchanged.
The integrated numerical contract, not algebra alone, decides admissibility.
"""
import torch
from torch.nn import functional as F


class OrderedContent(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, tokens):
        if weight.device.type!='cpu' or tokens.ndim!=2:
            raise ValueError('CPU [payload,vocabulary] weight and [lanes,time] IDs required')
        ctx.save_for_backward(tokens)
        ctx.shape=weight.shape
        # Return one independent output per event so each cotangent already
        # contains that event's factual, route and readout contributions.
        return tuple(F.embedding(tokens[:,k],weight.T) for k in range(tokens.shape[1]))

    @staticmethod
    def backward(ctx, *cotangents):
        (tokens,)=ctx.saved_tensors
        sample=next((g for g in cotangents if g is not None),None)
        if sample is None:return None,None
        gradient=sample.new_zeros(ctx.shape)
        # Native embedding first sums duplicate lanes within one time slice.
        # Accumulate those slices in original reverse temporal order. Compact
        # vocabulary IDs avoid materializing a vocabulary-size zero gradient
        # for every event; return one dense leaf gradient for existing AdamW.
        for k in range(tokens.shape[1]-1,-1,-1):
            g=cotangents[k]
            if g is None:continue
            unique,inverse=torch.unique(tokens[:,k],sorted=True,return_inverse=True)
            partial=torch.ops.aten.embedding_dense_backward(g,inverse,len(unique),-1,False)
            gradient[:,unique]+=partial.T
        return gradient,None


def ordered_content(weight,tokens):
    return OrderedContent.apply(weight,tokens)
