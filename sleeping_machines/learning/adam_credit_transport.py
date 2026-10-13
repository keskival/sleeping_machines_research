"""Matrix-free cotangent of a fixed-version clip/AdamW proposal.

All inputs are one declared complete optimizer vector. Global clipping retains
one cross-vector contraction: this helper does not make global Adam clockless.
"""
import torch

def adam_gradient_cotangent(gradient,first_moment,second_moment,parameter_cotangent,*,step,lr,beta1=.9,beta2=.999,adam_eps=1e-8,clip_radius=1.,clip_eps=1e-6,moment1_cotangent=None,moment2_cotangent=None):
    if step<1 or adam_eps<=0 or clip_radius<=0:raise ValueError('Valid Adam step and positive stabilizers required')
    if not (gradient.shape==first_moment.shape==second_moment.shape==parameter_cotangent.shape):raise ValueError('Complete vector shapes must agree')
    norm=gradient.norm();scale=(clip_radius/(norm+clip_eps)).clamp(max=1.)
    clipped=gradient*scale
    m=beta1*first_moment+(1-beta1)*clipped
    v=beta2*second_moment+(1-beta2)*clipped.square()
    mh=m/(1-beta1**step);vh=v/(1-beta2**step)
    if bool((vh<0).any()):raise ValueError('Nonnegative second moment required')
    root=vh.sqrt();safe_root=torch.where(root>0,root,torch.ones_like(root));denom=root+adam_eps
    # At a consistent untouched coordinate,mh=root=0, the composite Adam
    # proposal is differentiable even though sqrt alone has a cusp.
    if bool(((root==0)&(mh!=0)).any()):raise ValueError('Singular inconsistent moment coordinate')
    variance_term=torch.where(root>0,mh*(1-beta2)*clipped/((1-beta2**step)*safe_root*denom.square()),torch.zeros_like(root))
    diagonal=-lr*((1-beta1)/((1-beta1**step)*denom)-variance_term)
    cot=diagonal*parameter_cotangent
    if moment1_cotangent is not None:cot=cot+(1-beta1)*moment1_cotangent
    if moment2_cotangent is not None:cot=cot+2*(1-beta2)*clipped*moment2_cotangent
    if bool(scale<1):cot=scale*(cot-gradient*(gradient*cot).sum()/(norm*(norm+clip_eps)))
    return cot
