"""Connected finite-update field with forward-derived, learnable rank-two factors."""
import torch

class StructuredCredit(torch.nn.Module):
    def __init__(self,context_size,named,rank=2):
        super().__init__();self.rank=rank;self.names=list(named);self.shapes=[tuple(p.shape) for p in named.values()];self.lengths=[rank*(shape[0]+shape[1]) if len(shape)==2 else p.numel() for shape,p in zip(self.shapes,named.values())]
        self.net=torch.nn.Sequential(torch.nn.Linear(context_size,64),torch.nn.Tanh(),torch.nn.Linear(64,64),torch.nn.Tanh(),torch.nn.Linear(64,sum(self.lengths)))
        torch.nn.init.zeros_(self.net[-1].weight);torch.nn.init.zeros_(self.net[-1].bias)
    def forward(self,context,bases):
        pieces=self.net(context).split(self.lengths);updates=[]
        for name,shape,piece in zip(self.names,self.shapes,pieces):
            if len(shape)==2:
                rows,columns=shape;left=piece[:self.rank*rows].reshape(rows,self.rank);right=piece[self.rank*rows:].reshape(self.rank,columns)+bases[name]
                updates.append((left@right).flatten())
            else:updates.append(piece)
        return torch.cat(updates)

def unit(value):return value/value.square().sum().clamp_min(1e-16).sqrt()
def install_basis_hooks(actor):
    actor.input_bases={};actor.basis_handles=[]
    for name,module in actor.named_modules():
        if isinstance(module,torch.nn.Linear) and name.startswith(('gap','layers.')):
            def hook(module,args,name=name):
                if not getattr(actor,'collect_bases',False):return
                with torch.no_grad():
                    x=args[0].detach().reshape(-1,args[0].shape[-1]);mean=x.mean(0);rms=x.square().mean(0).clamp_min(1e-16).sqrt()
                    actor.input_bases[name+'.weight']=torch.stack([unit(mean),unit(rms)])
            actor.basis_handles.append(module.register_forward_pre_hook(hook))

def bases_for(actor,named):
    bases={}
    for name,parameter in named.items():
        if parameter.ndim!=2:continue
        if name in actor.input_bases:bases[name]=actor.input_bases[name]
        else:
            with torch.no_grad():value=parameter.detach();bases[name]=torch.stack([unit(value.mean(0)),unit(value.square().mean(0).clamp_min(1e-16).sqrt())])
    return bases
