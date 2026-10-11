"""Exact connected native actor with composed per-layer teaching packets."""
import torch
from check_native_packet_actor_update import native
from native_training_linearization import NativeTrainingLinearization
class ComposedTeachingPacket:
 def __init__(self,packets):self.packets=packets
 def apply(self,directions,*,context,version):
  for packet in self.packets:directions=packet.apply(directions,context=context,version=version)
  return directions
class ConnectedDeepTeachingActor(native.RaceTPP):
 def encode(self,t,m,mask):
  dt=torch.diff(t,dim=1,prepend=t[:,:1]).clamp_min(0)/self.scale;age=(t-t[:,:1])/self.scale
  h=self.embed(m)+self.gap(torch.stack([torch.log1p(dt),(dt>0).to(t.dtype),torch.log1p(age)],-1));packets=[]
  for layer in self.layers:
   packet=NativeTrainingLinearization.capture(layer,h,dt,context='taxi',version=self.credit_version);packets.append(packet);h=packet.forward_output
  self.packet=ComposedTeachingPacket(packets)
  return h,self.marks(h,m,dt)
