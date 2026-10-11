"""PSD forward-derived block covariance; includes correlations inside cached basis."""
import torch
from token_spectral_curvature import basis

@torch.no_grad()
def block_basis(embedding,rank=8):
    cache=basis(embedding,rank);b=cache['projected'];indices=torch.triu_indices(rank,rank)
    del cache['projected_square'];cache['pair_indices']=indices;cache['pair_products']=b[:,indices[0]]*b[:,indices[1]]
    return cache

@torch.no_grad()
def block_covariances(probability,receivers,dp,cache):
    b=cache['projected'];products=cache['pair_products'];r=torch.tensor(receivers,dtype=torch.long);p=probability[r]
    mean=probability@b;moment=probability@products;excluded_mean=(mean[None]-p[:,None]*b[r])/(1-p)[:,None];excluded_moment=(moment[None]-p[:,None]*products[r])/(1-p)[:,None]
    alternative_mean=mean[None]+dp[:,None]*(b[r]-excluded_mean);pairs=moment[None]+dp[:,None]*(products[r]-excluded_moment)
    rank=b.shape[1];cov=probability.new_zeros(len(r),rank,rank);i,j=cache['pair_indices'];cov[:,i,j]=pairs;cov[:,j,i]=pairs;cov=cov-alternative_mean[:,:,None]*alternative_mean[:,None,:]
    eigenvalues,vectors=torch.linalg.eigh((cov+cov.transpose(1,2))/2)
    return (vectors*eigenvalues.clamp_min(0)[:,None,:])@vectors.transpose(1,2)

@torch.no_grad()
def block_action(diagonal,covariance,u,displacement):
    lowdiag=torch.einsum('di,wij,dj->wd',u,covariance,u).clamp_min(0)
    ratio=torch.where(lowdiag>0,diagonal.clamp_min(0)/lowdiag.clamp_min(1e-30),torch.ones_like(lowdiag));scale=ratio.min(1).values.clamp(0,1)
    remainder=(diagonal-scale[:,None]*lowdiag).clamp_min(0)
    low=torch.einsum('wi,wij->wj',displacement@u,covariance)@u.T
    return remainder*displacement+scale[:,None]*low
