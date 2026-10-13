"""Matrix-free forward action tangent of actual global clip/ordinary Adam.

Complete-vector norm plus one contraction per action remains charged.
"""
import torch

def adam_action_tangent(gradient,first,second,directions,*,step,lr,beta1=.9,beta2=.999,adam_eps=1e-8,clip_radius=1.,clip_eps=1e-6):
    if directions.shape[0]!=gradient.numel():raise ValueError('Complete gradient/action shape mismatch')
    norm=gradient.norm();scale=(clip_radius/(norm+clip_eps)).clamp(max=1.)
    clipped=gradient*scale
    tangent=directions
    if bool(scale<1):
        tangent=scale*(directions-gradient[:,None]*((gradient@directions)/(norm*(norm+clip_eps)))[None,:])
    m=beta1*first+(1-beta1)*clipped
    v=beta2*second+(1-beta2)*clipped.square()
    if bool((v<0).any()):raise ValueError('Nonnegative variance required')
    root=(v/(1-beta2**step)).sqrt();safe=torch.where(root>0,root,torch.ones_like(root))
    mh=m/(1-beta1**step);denom=root+adam_eps
    if bool(((root==0)&(mh!=0)).any()):raise ValueError('Inconsistent zero-variance moment')
    variance=torch.where(root>0,mh*(1-beta2)*clipped/((1-beta2**step)*safe*denom.square()),torch.zeros_like(root))
    diagonal=-lr*((1-beta1)/((1-beta1**step)*denom)-variance)
    return diagonal[:,None]*tangent
