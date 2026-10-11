"""Shared native primal packet for exact input JVP and VJP; no dense Jacobian."""
import torch
from native_cached_jvp import NativeCachedJVP

class NativeCachedLinearization(NativeCachedJVP):
    @torch.no_grad()
    def transpose(self,cotangents,*,context,version):
        if context!=self.context or version!=self.version or self.parameter_versions!=tuple(p._version for p in self.layer.parameters()):raise ValueError('stale native linearization')
        if self.layer.training or self.layer.drop.training:raise ValueError('training mode changed')
        v=cotangents.detach()
        if v.ndim!=4 or tuple(v.shape[1:])!=tuple(self.shape) or not torch.isfinite(v).all():raise ValueError('invalid cotangent block')
        layer=self.layer;n=layer.n
        def norm_vjp(cotangent,data):
            normalized,inv,weight=data;weighted=cotangent*weight
            return inv*(weighted-weighted.mean(-1,keepdim=True)-normalized*(normalized*weighted).mean(-1,keepdim=True))
        pre2=norm_vjp(v,self.norm[1]);h=pre2+((pre2@layer.mlp[2].weight)*self.gelu_derivative)@layer.mlp[0].weight
        direct=norm_vjp(h,self.norm[0]);state=direct@layer.read.weight
        sr=v.new_zeros(v.shape[0],v.shape[1],n);si=torch.zeros_like(sr);result=direct.clone()
        for i in range(v.shape[2]-1,-1,-1):
            sr=sr+state[:,:,i,:n];si=si+state[:,:,i,n:]
            g=self.g[:,i]
            writes=torch.cat([sr*g,si*g],-1)
            gates=(sr*self.wr[:,i]+si*self.wi[:,i])*g*(1-g)
            result[:,:,i]+=writes@layer.write.weight+gates@layer.gate.weight
            sr,si=self.cr[:,i]*sr+self.ci[:,i]*si,-self.ci[:,i]*sr+self.cr[:,i]*si
        return result
