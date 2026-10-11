"""Shared forward teaching packet with the native producer graph connected.
Ordinary actor derivatives use native autograd; packet is for additional teaching.
"""
import torch
from check_native_packet_actor_update import PacketActor
from native_training_linearization import NativeTrainingLinearization
class ConnectedTeachingActor(PacketActor):
 def encode(self,t,m,mask):
  dt=torch.diff(t,dim=1,prepend=t[:,:1]).clamp_min(0)/self.scale;age=(t-t[:,:1])/self.scale
  x=self.embed(m)+self.gap(torch.stack([torch.log1p(dt),(dt>0).to(t.dtype),torch.log1p(age)],-1))
  assert len(self.layers)==1
  self.packet=NativeTrainingLinearization.capture(self.layers[0],x,dt,context='taxi',version=self.credit_version)
  h=self.packet.forward_output
  return h,self.marks(h,m,dt)
