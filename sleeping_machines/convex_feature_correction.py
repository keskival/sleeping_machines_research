"""TRAIN-only convex correction fit over frozen native temporal features.

Does not train/replace the temporal core. The fixed dense/adaptive normalized
readout is affine-log-softmax along a feature correction, making this subproblem
convex. A feasible zero correction supplies an empirical objective certificate,
not a held-out gain or a certificate for joint feature learning.
"""
import copy
import math
import torch
from torch import nn
from .token_readout import TokenReadout


def fit_feature_correction(readout, parent_features, branch_features, targets, *,
                           l2=1e-3, iterations=32, history_size=8):
    if not math.isfinite(l2) or l2 <= 0 or iterations <= 0 or history_size <= 0:
        raise ValueError('Positive finite regularization and solver limits required')
    if getattr(readout.nll, '__func__', None) is not TokenReadout.nll:
        raise ValueError('Only the declared affine normalized readout is supported')
    output=readout.output
    if isinstance(output, nn.AdaptiveLogSoftmaxWithLoss):
        if not isinstance(output.head,nn.Linear) or any(not isinstance(layer,nn.Linear) for tail in output.tail for layer in tail):
            raise ValueError('Adaptive branches must be affine')
    elif not isinstance(output,nn.Linear):
        raise ValueError('Affine dense or adaptive readout required')
    if (parent_features.ndim<2 or parent_features.shape[:-1]!=branch_features.shape[:-1]
            or parent_features.shape[:-1]!=targets.shape or targets.numel()==0
            or targets.dtype!=torch.long or parent_features.device.type!='cpu'
            or parent_features.dtype!=branch_features.dtype or parent_features.device!=branch_features.device):
        raise ValueError('Compatible nonempty CPU feature/target arrays required')
    if not bool(torch.isfinite(parent_features).all()) or not bool(torch.isfinite(branch_features).all()):
        raise ValueError('Finite frozen features required')
    frozen=copy.deepcopy(readout).eval()
    for p in frozen.parameters():p.requires_grad_(False)
    parent=parent_features.detach();branch=branch_features.detach()
    weight=nn.Parameter(parent.new_zeros(parent.shape[-1],branch.shape[-1]))
    with torch.no_grad():baseline=float(frozen.nll(parent,targets).mean())
    if not math.isfinite(baseline):raise ValueError('Finite baseline loss required')
    best_loss=baseline;best_weight=weight.detach().clone();calls=0
    optimizer=torch.optim.LBFGS([weight],lr=1.,max_iter=iterations,history_size=history_size,line_search_fn='strong_wolfe')
    def closure():
        nonlocal best_loss,best_weight,calls
        optimizer.zero_grad()
        loss=frozen.nll(parent+branch@weight.T,targets).mean()+.5*l2*weight.square().sum()
        if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite correction objective')
        loss.backward();calls+=1
        if float(loss.detach())<best_loss:
            best_loss=float(loss.detach());best_weight=weight.detach().clone()
        return loss
    optimizer.step(closure)
    with torch.no_grad():
        fitted=float(frozen.nll(parent+branch@best_weight.T,targets).mean())
        objective=fitted+.5*l2*float(best_weight.square().sum())
    tolerance=1e-6 if parent.dtype==torch.float32 else 1e-10
    if objective>baseline+tolerance:raise RuntimeError('Feasible-zero empirical certificate failed')
    return dict(weight=best_weight,baseline_nll=baseline,fitted_nll=fitted,
                regularized_objective=objective,l2=l2,closure_calls=calls,
                targets=targets.numel(),scope='Frozen TRAIN-panel convex subproblem; no DEV/test selection, feature learning or generalization claim')
