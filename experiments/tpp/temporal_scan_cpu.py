"""Exact CPU affine temporal scan; native math, one thread, explicit carry.
Custom backward is first-order. Higher-order meta-gradients need a separately
validated differentiable backward and are not supported by this backend.
"""
import hashlib
import os
from pathlib import Path

import torch
from torch.autograd.function import once_differentiable

ROOT=Path(__file__).resolve().parents[2]
_EXTENSION=None


def extension():
    global _EXTENSION
    if _EXTENSION is None:
        from torch.utils.cpp_extension import load
        os.environ['MAX_JOBS']='1'
        source=Path(__file__).with_suffix('.cpp')
        tag=hashlib.sha256(source.read_bytes()).hexdigest()[:12]
        directory=ROOT/'.git/temporal-scan-build'/tag;directory.mkdir(parents=True,exist_ok=True)
        load(name='temporal_scan_'+tag,sources=[str(source)],
            extra_cflags=['-O2','-ffp-contract=off'],with_cuda=False,
            build_directory=str(directory),verbose=True,is_python_module=False)
        _EXTENSION=torch.ops.temporal_scan_cpu
    return _EXTENSION


class Scan(torch.autograd.Function):
    @staticmethod
    def forward(ctx,cr,ci,wr,wi,initial):
        out=extension().forward(cr,ci,wr,wi,initial)
        ctx.save_for_backward(cr,ci,initial,out)
        return out
    @staticmethod
    @once_differentiable
    def backward(ctx,upstream):
        cr,ci,initial,out=ctx.saved_tensors
        return tuple(extension().backward(cr,ci,initial,out,upstream.contiguous()))


def scan(cr,ci,wr,wi,initial=None):
    if initial is None:initial=cr.new_zeros(cr.shape[0],2*cr.shape[-1])
    return Scan.apply(cr.contiguous(),ci.contiguous(),wr.contiguous(),wi.contiguous(),initial.contiguous())


def reference(cr,ci,wr,wi,initial=None):
    if initial is None:initial=cr.new_zeros(cr.shape[0],2*cr.shape[-1])
    r,i=initial.chunk(2,-1);outs=[]
    for t in range(cr.shape[1]):
        r,i=cr[:,t]*r-ci[:,t]*i+wr[:,t],ci[:,t]*r+cr[:,t]*i+wi[:,t]
        outs.append(torch.cat((r,i),-1))
    return torch.stack(outs,1)


def layer_forward(layer,u,dt):
    # Unchanged native projections, gates, read, norms and feedforward network.
    w=layer.write(u);g=layer.gate(u).sigmoid();wr,wi=w[...,:layer.n]*g,w[...,layer.n:]*g
    decay=torch.exp(-layer.log_rate.exp()*dt.unsqueeze(-1));angle=layer.freq*dt.unsqueeze(-1)
    z=scan(decay*angle.cos(),decay*angle.sin(),wr,wi)
    h=layer.norm1(u+layer.drop(layer.read(z)))
    return layer.norm2(h+layer.drop(layer.mlp(h)))
