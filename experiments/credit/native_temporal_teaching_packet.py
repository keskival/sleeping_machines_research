"""Exact content/time transport from one native primal; eval mode only."""
import torch
from native_cached_linearization import NativeCachedLinearization
class NativeTemporalTeachingPacket(NativeCachedLinearization):
 @classmethod
 @torch.no_grad()
 def capture(cls,layer,u,dt,*,context,version):
  packet=super().capture(layer,u,dt,context=context,version=version)
  zr=u.new_zeros(u.shape[0],layer.n);zi=torch.zeros_like(zr);real=[];imag=[];rate=layer.log_rate.exp();freq=layer.freq
  for i in range(u.shape[1]):
   cr,ci=packet.cr[:,i],packet.ci[:,i];dcr=-rate*cr-freq*ci;dci=-rate*ci+freq*cr
   real.append(dcr*zr-dci*zi);imag.append(dci*zr+dcr*zi)
   zr,zi=cr*zr-ci*zi+packet.g[:,i]*packet.wr[:,i],ci*zr+cr*zi+packet.g[:,i]*packet.wi[:,i]
  packet.time_real=torch.stack(real,1);packet.time_imag=torch.stack(imag,1);return packet
 @torch.no_grad()
 def apply_with_time(self,directions,time_directions,*,context,version):
  base=super().apply(directions,context=context,version=version)
  if tuple(time_directions.shape)!=tuple(directions.shape[:-1]) or not torch.isfinite(time_directions).all():raise ValueError('invalid elapsed-time directions')
  v=directions;w=time_directions;zr=v.new_zeros(v.shape[0],v.shape[1],self.layer.n);zi=torch.zeros_like(zr);states=[]
  for i in range(v.shape[2]):
   zr,zi=self.cr[:,i]*zr-self.ci[:,i]*zi+self.time_real[:,i]*w[:,:,i,None],self.ci[:,i]*zr+self.cr[:,i]*zi+self.time_imag[:,i]*w[:,:,i,None]
   states.append(torch.cat([zr,zi],-1))
  h=torch.stack(states,2)@self.layer.read.weight.T
  def norm_jvp(tangent,data):
   normalized,inv,weight=data;return inv*(tangent-tangent.mean(-1,keepdim=True)-normalized*(normalized*tangent).mean(-1,keepdim=True))*weight
  h=norm_jvp(h,self.norm[0]);mlp=(h@self.layer.mlp[0].weight.T)*self.gelu_derivative
  return base+norm_jvp(h+mlp@self.layer.mlp[2].weight.T,self.norm[1])
 @torch.no_grad()
 def transpose_with_time(self,cotangents,*,context,version):
  content=super().transpose(cotangents,context=context,version=version);v=cotangents;layer=self.layer
  def norm_vjp(cotangent,data):
   normalized,inv,weight=data;weighted=cotangent*weight;return inv*(weighted-weighted.mean(-1,keepdim=True)-normalized*(normalized*weighted).mean(-1,keepdim=True))
  pre2=norm_vjp(v,self.norm[1]);h=pre2+((pre2@layer.mlp[2].weight)*self.gelu_derivative)@layer.mlp[0].weight
  state=norm_vjp(h,self.norm[0])@layer.read.weight;sr=v.new_zeros(v.shape[0],v.shape[1],layer.n);si=torch.zeros_like(sr);elapsed=v.new_zeros(v.shape[:-1])
  for i in range(v.shape[2]-1,-1,-1):
   sr=sr+state[:,:,i,:layer.n];si=si+state[:,:,i,layer.n:]
   elapsed[:,:,i]=(sr*self.time_real[:,i]+si*self.time_imag[:,i]).sum(-1)
   sr,si=self.cr[:,i]*sr+self.ci[:,i]*si,-self.ci[:,i]*sr+self.cr[:,i]*si
  return content,elapsed
 def cached_floats(self):return super().cached_floats()+self.time_real.numel()+self.time_imag.numel()
