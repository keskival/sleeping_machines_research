"""Exact four-clock parameter-step Gram without a dense n-by-action tangent.

Disjoint sparse action supports; global clip norm and a scalar contraction
remain. This does not make the optimizer clockless.
"""
import torch

def sparse_adam_metric(g,m,v,supports,*,step,lr,beta1=.9,beta2=.999,adam_eps=1e-8,clip_radius=1.,clip_eps=1e-6):
    norm=g.norm();k=(clip_radius/(norm+clip_eps)).clamp(max=1.)
    z=k*g;first=beta1*m+(1-beta1)*z;second=beta2*v+(1-beta2)*z.square()
    root=(second/(1-beta2**step)).sqrt();safe=torch.where(root>0,root,torch.ones_like(root));mh=first/(1-beta1**step);denom=root+adam_eps
    if bool((second<0).any()) or bool(((root==0)&(mh!=0)).any()):raise ValueError('Invalid moments')
    term=torch.where(root>0,mh*(1-beta2)*z/((1-beta2**step)*safe*denom.square()),torch.zeros_like(root))
    d=-lr*((1-beta1)/((1-beta1**step)*denom)-term)
    # J = U - w a^T, U=k D B, w=k D g. B supports disjoint.
    w=k*d*g;diagonal=[];cross=[];contraction=[];seen=set()
    for indices,values in supports:
        entries=set(indices.tolist())
        if seen&entries:raise ValueError('Action supports must be disjoint')
        seen.update(entries)
        u=k*d[indices]*values
        diagonal.append(u.square().sum());cross.append((u*w[indices]).sum())
        contraction.append((g[indices]*values).sum())
    diagonal=torch.stack(diagonal);cross=torch.stack(cross)
    a=torch.stack(contraction)/(norm*(norm+clip_eps)) if bool(k<1) else torch.zeros_like(cross)
    return torch.diag(diagonal)-cross[:,None]*a[None,:]-a[:,None]*cross[None,:]+w.square().sum()*a[:,None]*a[None,:]
