"""One-dimensional convex FIT calibration preserving native processing.

q=softmax(log prior + gain*(logits-center)); gain>=0. No change to representations,
races, memory or timing. Fitting/evaluating on the same data is not generalization.
"""
import torch
from torch.nn import functional as F


def fit_prior_centered(logits,labels,max_gain=256.):
    z=logits.detach().to(torch.float64);y=labels.to(z.device)
    prior=torch.bincount(y,minlength=z.shape[1]).to(z.dtype)/len(y)
    if not (prior>0).all():raise ValueError('all declared classes need FIT support or an explicit smoothing policy')
    center=z.mean(0);residual=z-center;log_prior=prior.log()
    association=(residual[torch.arange(len(y),device=y.device),y]-(residual*prior).sum(1)).mean()
    def derivative(gain):
        q=(log_prior+gain*residual).softmax(-1)
        return float(((q*residual).sum(1)-residual[torch.arange(len(y),device=y.device),y]).mean())
    if association<=0:gain=0.
    else:
        lo,hi=0.,1.
        while hi<max_gain and derivative(hi)<0:hi=min(max_gain,hi*2)
        if derivative(hi)<0:gain=hi
        else:
            for _ in range(64):
                mid=(lo+hi)/2
                if derivative(mid)<0:lo=mid
                else:hi=mid
            gain=(lo+hi)/2
    calibrated=log_prior+gain*residual
    return dict(center=center,log_prior=log_prior,gain=gain,
        nll=float(F.cross_entropy(calibrated,y)),accuracy=float((calibrated.argmax(-1)==y).double().mean()),
        association=float(association),calibrated_association=float(gain*association),
        prior_nll=float(-(prior*log_prior).sum()),gain_at_cap=gain==max_gain)


def linear_parameters(weight,bias,calibration):
    gain=calibration['gain']
    return gain*weight,(calibration['log_prior']+gain*(bias.to(torch.float64)-calibration['center'])).to(bias.dtype)
