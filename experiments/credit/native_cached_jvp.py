"""Exact input tangents from one native TemporalMemoryLayer primal capture.
Eval-mode fixed elapsed times, input directions only. No hard-route derivatives.
"""
import math
import torch

class NativeCachedJVP:
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
        try:output=layer(u,dt).detach().clone()
        finally:
            for handle in handles:handle.remove()
        self=cls();self.layer=layer;self.context=context;self.version=version;self.shape=u.shape
        self.parameter_versions=tuple(p._version for p in layer.parameters());self.output=output
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

    @torch.no_grad()
    def apply(self,directions,*,context,version):
        if context!=self.context or version!=self.version or self.parameter_versions!=tuple(p._version for p in self.layer.parameters()):raise ValueError('stale native linearization')
        if self.layer.training or self.layer.drop.training:raise ValueError('training mode changed')
        v=directions.detach()
        if v.ndim!=4 or tuple(v.shape[1:])!=tuple(self.shape) or not torch.isfinite(v).all():raise ValueError('invalid tangent block')
        layer=self.layer;n=layer.n
        dw=v@layer.write.weight.T;dg=(v@layer.gate.weight.T)*self.g*(1-self.g)
        wr=dw[...,:n]*self.g+self.wr*dg;wi=dw[...,n:]*self.g+self.wi*dg
        zr=v.new_zeros(v.shape[0],v.shape[1],n);zi=torch.zeros_like(zr);states=[]
        for i in range(v.shape[2]):
            zr,zi=self.cr[:,i]*zr-self.ci[:,i]*zi+wr[:,:,i],self.cr[:,i]*zi+self.ci[:,i]*zr+wi[:,:,i]
            states.append(torch.cat([zr,zi],-1))
        state=torch.stack(states,2);h=v+state@layer.read.weight.T
        def norm_jvp(tangent,data):
            normalized,inv,weight=data
            return inv*(tangent-tangent.mean(-1,keepdim=True)-normalized*(normalized*tangent).mean(-1,keepdim=True))*weight
        h=norm_jvp(h,self.norm[0]);mlp=(h@layer.mlp[0].weight.T)*self.gelu_derivative
        return norm_jvp(h+mlp@layer.mlp[2].weight.T,self.norm[1])

    def cached_floats(self):
        tensors=[self.output,self.g,self.wr,self.wi,self.cr,self.ci,self.gelu_derivative]
        tensors += [value for group in self.norm for value in group[:2]]
        return sum(t.numel() for t in tensors)
