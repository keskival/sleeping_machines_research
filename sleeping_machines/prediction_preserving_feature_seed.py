"""Cross-credit direction for a fully contained zero-output feature branch.

Returns C^T u without mutating parameters or drawing randomness. The caller
must separately prove that zero output makes prediction independent of the
feature parameters, including clocks, state, context and all downstream uses.
This is a finite issued-loss derivative, not a full stochastic route teacher.
"""
import torch


def cross_credit_direction(loss,output_parameters,feature_parameters,probes):
    outputs=tuple(output_parameters);features=tuple(feature_parameters);probes=tuple(probes)
    if loss.ndim!=0 or not loss.requires_grad or not bool(torch.isfinite(loss)) or not outputs or not features or len(outputs)!=len(probes):
        raise ValueError('Scalar loss, nonempty parameter groups and one probe per output required')
    if len({id(p) for p in outputs+features})!=len(outputs)+len(features):
        raise ValueError('Output and feature leaves must be distinct and canonical')
    for parameter in outputs:
        if not parameter.is_leaf or not parameter.requires_grad or not bool(torch.isfinite(parameter).all()) or bool(parameter.detach().ne(0).any()):
            raise ValueError('Finite zero output maps with live derivative required')
    for parameter in features:
        if not parameter.is_leaf or not parameter.requires_grad or not bool(torch.isfinite(parameter).all()):
            raise ValueError('Finite trainable feature leaves required')
    for parameter,probe in zip(outputs,probes):
        if probe.shape!=parameter.shape or probe.dtype!=parameter.dtype or probe.device!=parameter.device or not bool(torch.isfinite(probe).all()):
            raise ValueError('Probe schema mismatch or nonfinite probe')
    gradients=torch.autograd.grad(loss,outputs,create_graph=True,retain_graph=True)
    scalar=sum((gradient*probe.detach()).sum() for gradient,probe in zip(gradients,probes))
    directions=(torch.autograd.grad(scalar,features,allow_unused=True,retain_graph=True)
                if scalar.requires_grad else (None,)*len(features))
    directions=tuple(torch.zeros_like(p) if d is None else d.detach() for p,d in zip(features,directions))
    if any(not bool(torch.isfinite(d).all()) for d in directions):
        raise FloatingPointError('Nonfinite cross-credit direction')
    return directions


def bounded_seed_direction(directions,radius):
    """Euclidean trust-radius scaling; does not update any actor or optimizer."""
    directions=tuple(directions)
    if not directions or not isinstance(radius,(float,int)) or not 0<radius<float('inf'):
        raise ValueError('Nonempty directions and finite positive radius required')
    if any(not bool(torch.isfinite(d).all()) for d in directions):raise ValueError('Finite directions required')
    norm=sum(float(d.double().square().sum()) for d in directions)**.5
    scale=radius/norm if norm else 0.
    return tuple(d*scale for d in directions)
