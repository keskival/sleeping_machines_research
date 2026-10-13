"""Typed clock-mode optimizer receipts and a native bounded residual critic."""
import copy
from dataclasses import dataclass
import torch
import numpy as np
from native_market_credit import NativeCarry

@dataclass
class ReceiptCarry:
    parent: object
    receipt: object
    rng: object
    def detach(self):
        return ReceiptCarry(self.parent.detach() if self.parent is not None else None,self.receipt.detach() if self.receipt is not None else None,self.rng.clone())


def descriptor(model,opt,grads):
    by=dict(zip((name for name,_ in model.named_parameters()),grads));rows=[];steps=[]
    for j,layer in enumerate(model.layers):
        pair=[]
        for key in ('log_rate','freq'):
            name=f'layers.{j}.{key}';p=dict(model.named_parameters())[name];state=opt.state[p]
            pair.append(torch.stack((p.detach(),by[name].detach(),state.get('exp_avg',torch.zeros_like(p)).detach(),state.get('exp_avg_sq',torch.zeros_like(p)).detach()),-1));steps.append(int(state.get('step',0))+1)
        rows.append(torch.stack(pair,0))
    assert len(set(steps))==1
    group=opt.param_groups[0]
    return dict(clock=torch.stack(rows),gradient_square=sum(g.detach().square().sum() for g in grads),step=steps[0],betas=group['betas'],eps=group['eps'],lr=group['lr'])


def events(features,observed,operator,baseline,scale):
    clock=operator['clock'];layers,legs,modes,_=clock.shape;assert legs==2 and layers*modes<=32
    g=clock[...,1];correction=(baseline*scale).reshape(layers,legs,1)/np.sqrt(modes)
    issued=g+correction;norm=(operator['gradient_square']+(issued.square()-g.square()).sum()).clamp_min(0).sqrt();k=(1/(norm+1e-6)).clamp(max=1.)
    z=k*issued;b1,b2=operator['betas'];step=operator['step']
    first=b1*clock[...,2]+(1-b1)*z;second=b2*clock[...,3]+(1-b2)*z.square()
    root=(second/(1-b2**step)).sqrt();safe=torch.where(root>0,root,torch.ones_like(root));mh=first/(1-b1**step);denom=root+operator['eps']
    variance=torch.where(root>0,mh*(1-b2)*z/((1-b2**step)*safe*denom.square()),torch.zeros_like(root))
    signed=-operator['lr']*((1-b1)/((1-b1**step)*denom)-variance)
    tokens=[];width=features.shape[-1];state_width=width//3
    for layer in range(layers):
        for mode in range(modes):
            token=features.new_zeros(width+8);token[192+layer*modes+mode]=1.
            values=torch.cat((clock[layer,:,mode,:].reshape(-1),signed[layer,:,mode],features[0,state_width+2*modes*layer+mode:state_width+2*modes*layer+mode+1],features[0,state_width+2*modes*layer+modes+mode:state_width+2*modes*layer+modes+mode+1],baseline[2*layer:2*layer+2],k.reshape(1)))
            token[224:224+values.numel()]=values.asinh();token[width+1+4+layer]=1.
            tokens.append(token)
    return torch.cat((observed,torch.stack(tokens)[None]),1)


class ReceiptCredit(torch.nn.Module):
    def __init__(self,base,scale):
        super().__init__();self.parent=copy.deepcopy(base);self.parent.requires_grad_(False)
        self.receipt=copy.deepcopy(base);self.core=self.receipt.core
        torch.nn.init.zeros_(self.core.head.weight);torch.nn.init.zeros_(self.core.head.bias)
        self.register_buffer('scale',torch.as_tensor(scale,dtype=torch.float64));self.last_parent=None
    def forward(self,features,observed,carry,operator):
        parent_carry=None if carry is None else carry.parent
        receipt_carry=None if carry is None else carry.receipt
        self.parent.core.train(False)
        with torch.no_grad():baseline,parent_carry=self.parent(features,observed,parent_carry)
        self.last_parent=baseline[0].detach()
        tokens=events(features,observed,operator,self.last_parent,self.scale)
        rng=torch.Generator().manual_seed(99173).get_state() if carry is None else carry.rng
        # Private native receipt clocks do not consume the factual parent's tape.
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(rng)
            residual,receipt_carry=self.receipt(features,tokens,receipt_carry)
            next_rng=torch.get_rng_state().clone()
        return baseline+.5*residual.tanh(),ReceiptCarry(parent_carry,receipt_carry,next_rng)
