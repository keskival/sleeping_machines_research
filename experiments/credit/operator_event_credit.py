"""Current-operator recursive credit, plus future-trained temporal/event corrections."""
import math
import torch
from batched_event_parameter_credit import EventOptimizerCredit as Base
from event_parameter_credit import packet as event_packet

def packet(actor,c,loss,t):
    p=event_packet(actor,c,loss,t);p['operators']={n:v.detach().clone() for n,v in actor.named_parameters() if n.startswith(('embed.','gap.','layers.'))};return p

def norm_vjp(c,x,weight,eps=1e-5):
    centered=x-x.mean(-1,keepdim=True);inverse=centered.square().mean(-1,keepdim=True).add(eps).rsqrt();z=centered*inverse;scaled=c*weight
    return inverse*(scaled-scaled.mean(-1,keepdim=True)-z*(scaled*z).mean(-1,keepdim=True)),c*z,c

def local_vjp(u,z,dt,operators,c):
    """Exact current-event adjoint with incoming temporal state held fixed."""
    W=lambda n:operators[n+'.weight'];b=lambda n:operators[n+'.bias'];N=u.shape[-1]//2
    s1=u+z@W('read').T+b('read');h=torch.nn.functional.layer_norm(s1,(u.shape[-1],),W('norm1'),b('norm1'));a=h@W('mlp.0').T+b('mlp.0');activation=torch.nn.functional.gelu(a);s2=h+activation@W('mlp.2').T+b('mlp.2');cs2,gn2w,gn2b=norm_vjp(c,s2,W('norm2'));ca=(cs2@W('mlp.2'))*(.5*(1+torch.erf(a/math.sqrt(2)))+a*torch.exp(-a.square()/2)/math.sqrt(2*math.pi));ch=cs2+ca@W('mlp.0');cs1,gn1w,gn1b=norm_vjp(ch,s1,W('norm1'));cz=cs1@W('read');write=u@W('write').T+b('write');gate=torch.sigmoid(u@W('gate').T+b('gate'));cw=cz*torch.cat([gate,gate],-1);cg=(cz[:,:N]*write[:,:N]+cz[:,N:]*write[:,N:])*gate*(1-gate);cu=cs1+cw@W('write')+cg@W('gate');rotated=z-write*torch.cat([gate,gate],-1);rate=operators['log_rate'].exp();clockrate=-dt[:,None]*rate*(cz[:,:N]*rotated[:,:N]+cz[:,N:]*rotated[:,N:]);clockfreq=dt[:,None]*(-cz[:,:N]*rotated[:,N:]+cz[:,N:]*rotated[:,:N])
    Q=len(u);grads={}
    for name,q,x in [('read',cs1,z),('mlp.0',ca,h),('mlp.2',cs2,activation),('write',cw,u),('gate',cg,u)]:grads[name+'.weight']=q.T@x/Q;grads[name+'.bias']=q.mean(0)
    grads.update({'norm1.weight':gn1w.mean(0),'norm1.bias':gn1b.mean(0),'norm2.weight':gn2w.mean(0),'norm2.bias':gn2b.mean(0),'log_rate':clockrate.mean(0),'freq':clockfreq.mean(0)})
    return cu,grads

class EventOptimizerCredit(Base):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.context_handle=self.local.register_forward_hook(self.capture)
    def capture(self,module,args,output):self.local_contexts=output
    def forward(self,p,forced=None,pool=None):
        residual=super().forward(p,forced,pool);ids=self.last_selected;Q=len(ids);length=len(p['cotangent']);operators=p['operators'];message=p['cotangent'][ids];anchors={};depth=len(p['states'])-1
        for layer in range(depth-1,-1,-1):
            prefix=f'layers.{layer}.';local={n[len(prefix):]:v for n,v in operators.items() if n.startswith(prefix)};message=message+.01*self.decoders['norm2_bias'](self.local_contexts[layer+1]);message,gradients=local_vjp(p['inputs'][prefix+'write.weight'][ids],p['inputs'][prefix+'read.weight'][ids],p['dt'][ids],local,message)
            anchors.update({prefix+n:length*g for n,g in gradients.items()})
        anchors['gap.weight']=length*message.T@p['inputs']['gap.weight'][ids]/Q;anchors['gap.bias']=length*message.mean(0);anchors['embed.weight']=message.new_zeros(operators['embed.weight'].shape).index_add(0,p['marks'][ids],message)*length/Q
        return residual+torch.cat([anchors[n].flatten() for n in self.names])
