"""Connected learning-side critic; future observations never enter the actor."""
from dataclasses import dataclass
import torch
from torch import nn
from shared_prefix_receiver_credit import SharedPrefixCredit, prior


@dataclass
class FutureContext:
    value:torch.Tensor
    forward_version:int
    critic_version:int
    future_value:torch.Tensor
    horizon:int


def observed_future(t,m):
    assert t.shape==m.shape and t.shape[1]==8
    gaps=(t[:,4:]-t[:,3:-1]).clamp_min(0).log1p().unsqueeze(-1)
    marks=torch.nn.functional.one_hot(m[:,4:],8).to(t.dtype)
    return torch.cat((gaps,marks),-1)


class RevealedFutureCredit(SharedPrefixCredit):
    def __init__(self,M,U,R=8):
        super().__init__(M,U,R)
        self.future=nn.GRU(9,24,batch_first=True)
        self.future_projection=nn.Linear(24,38)
        nn.init.zeros_(self.future_projection.weight);nn.init.zeros_(self.future_projection.bias)

    def encode_packet(self,x,future,forward_version,critic_version):
        if future.shape!=(len(x),4,9):raise ValueError('Four revealed exogenous events required')
        base=super().encode_context(x,forward_version,critic_version)
        f=self.future(future.tanh())[0][:,-1]
        return FutureContext(base.value,forward_version,critic_version,self.future_projection(f),4)

    def values(self,activation,routes,actions,forward_version,critic_version):
        if not isinstance(activation,FutureContext) or activation.horizon!=4:
            raise ValueError('Learning-side values require a revealed future packet')
        if (activation.forward_version,activation.critic_version)!=(forward_version,critic_version):
            raise ValueError('Stale context activation')
        B,K,T,W=routes.shape;assert W==8 and actions.shape[:2]==(B,K)
        r=self.receiver(routes.reshape(B*K,T,8).tanh())[0][:,-1]
        c=activation.value[:,None].expand(-1,K,-1).reshape(B*K,31)
        f=activation.future_value[:,None].expand(-1,K,-1).reshape(B*K,38)
        v=self.head(torch.cat((c,r),-1)+f).reshape(B*K,self.R,6+2*self.M)
        raw=v[:,:,:4];u=self.U*raw/(1+raw.norm(dim=-1,keepdim=True))
        p=v[:,:,4:4+self.M].sigmoid();logw=v[:,:,4+self.M:4+2*self.M].log_softmax(-1)
        b=v[:,:,-2].sigmoid();mix=v[:,:,-1].softmax(-1)
        return (prior.decode(u,p,logw,b,actions.reshape(B*K,-1,4))*mix[:,:,None]).sum(1).reshape(B,K,-1)


def scores(model,x,future,version):
    packet=model.encode_packet(x[:,0,:,:30],future,0,version)
    actions=prior.base.actions()[None,None].expand(len(x),x.shape[1],-1,-1)
    return model.values(packet,x[:,:,:,30:],actions,0,version)


def contrasts(model,x,future,actions,references,version):
    packet=model.encode_packet(x[:,0,:,:30],future,0,version)
    index=torch.tensor(list(zip(actions,references)))
    routes=x[torch.arange(len(x))[:,None],index//9,:,30:]
    messages=prior.base.actions()[index%9][:,:,None]
    values=model.values(packet,routes,messages,0,version).squeeze(-1)
    return values[:,0]-values[:,1]
