"""Forward-derived PSD low-rank covariance with exact diagonal and charged basis."""
import torch

@torch.no_grad()
def basis(embedding,rank=8):
    if not 0<=rank<=embedding.shape[1]:raise ValueError('Invalid spectral rank')
    mean=embedding.mean(0);covariance=embedding.T@embedding/len(embedding)-mean[:,None]*mean[None]
    eigenvalues,vectors=torch.linalg.eigh(covariance)
    u=vectors[:,-rank:] if rank else vectors[:,:0]
    projected=embedding@u
    return dict(u=u,projected=projected,projected_square=projected.square(),eigenvalues=eigenvalues[-rank:] if rank else eigenvalues[:0])

@torch.no_grad()
def covariance_action(diagonal,variances,u,displacement):
    if variances.shape[1]!=u.shape[1] or diagonal.shape!=displacement.shape:raise ValueError('Covariance dimensions')
    variances=variances.clamp_min(0);lowdiag=variances@u.square().T
    ratio=torch.where(lowdiag>0,diagonal.clamp_min(0)/lowdiag.clamp_min(1e-30),torch.ones_like(lowdiag))
    scale=ratio.min(1).values.clamp(0,1);weights=variances*scale[:,None]
    remainder=(diagonal-weights@u.square().T).clamp_min(0)
    return remainder*displacement+((displacement@u)*weights)@u.T

@torch.no_grad()
def projected_variances(probability,receivers,dp,cache):
    b=cache['projected'];second=cache['projected_square'];mean=probability@b;moment=probability@second
    r=torch.tensor(receivers,dtype=torch.long);p=probability[r]
    excluded_mean=(mean[None]-p[:,None]*b[r])/(1-p)[:,None]
    excluded_moment=(moment[None]-p[:,None]*second[r])/(1-p)[:,None]
    alternative_mean=mean[None]+dp[:,None]*(b[r]-excluded_mean)
    alternative_moment=moment[None]+dp[:,None]*(second[r]-excluded_moment)
    return (alternative_moment-alternative_mean.square()).clamp_min(0)
