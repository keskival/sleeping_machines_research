"""Exact input tangents from one native TemporalMemoryLayer primal capture.
Eval-mode fixed elapsed times, input directions only; retained forward parameter graph. No hard-route derivatives.
"""
import math
import torch

from native_cached_linearization import NativeCachedLinearization

class NativeTrainingLinearization(NativeCachedLinearization):
    @classmethod
    @torch.no_grad()
    def capture(cls,layer,u,dt,*,context,version):
        if layer.training or layer.drop.training or layer.mlp[1].approximate!='none':raise ValueError('fixed eval-mode exact GELU required')
        if u.ndim!=3 or dt.shape!=u.shape[:2]:raise ValueError('invalid native input')
        cache={};handles=[]
        for name,module in [('write',layer.write),('gate',layer.gate),('norm1',layer.norm1),('mlp0',layer.mlp[0]),('norm2',layer.norm2)]:
            def hook(mod,inputs,output,name=name):
                cache[name]=(inputs[0].detach().clone(),output.detach().clone())
            handles.append(module.register_forward_hook(hook))
        try:
            with torch.enable_grad():output=layer(u,dt)
        finally:
            for handle in handles:handle.remove()
        self=cls();self.layer=layer;self.context=context;self.version=version;self.shape=u.shape
        self.parameter_versions=tuple(p._version for p in layer.parameters());self.output=output.detach().clone();self.forward_output=output
        n=layer.n;w=cache['write'][1];g=torch.sigmoid(cache['gate'][1]);self.g=g;self.wr=w[...,:n];self.wi=w[...,n:]
        decay=torch.exp(-layer.log_rate.exp()*dt.unsqueeze(-1));angle=layer.freq*dt.unsqueeze(-1)
        self.cr=decay*torch.cos(angle);self.ci=decay*torch.sin(angle)
        self.norm=[]
        for name,normalizer in [('norm1',layer.norm1),('norm2',layer.norm2)]:
            primal=cache[name][0];center=primal-primal.mean(-1,keepdim=True)
            inv=torch.rsqrt(center.square().mean(-1,keepdim=True)+normalizer.eps)
            self.norm.append((center*inv,inv,normalizer.weight.detach()))
        primal=cache['mlp0'][1]
        self.gelu_derivative=.5*(1+torch.erf(primal/math.sqrt(2)))+primal*torch.exp(-primal.square()/2)/math.sqrt(2*math.pi)
        return self

