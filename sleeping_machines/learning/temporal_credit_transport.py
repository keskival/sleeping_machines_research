"""Fixed-version temporal-state adjoint; no dense transition or history tape.

Only the direct decay/rotation path is included. Incoming writes, hard-route
alternatives and parameter/optimizer paths require their own credit packets.
"""
import torch


def transport_cotangent(cotangent, log_rate, frequency, elapsed):
    """Pull [real, imaginary] state credit across elapsed physical time.

    Caller owns parameter versions and complete dependency validation. Multiple
    gaps may be summed only while the clock parameters remain fixed and no
    other state-dependent transition lies on the declared path.
    """
    n = log_rate.numel()
    if frequency.shape != log_rate.shape or cotangent.shape[-1] != 2 * n:
        raise ValueError('Clock and state-interface dimensions differ')
    elapsed = torch.as_tensor(elapsed, dtype=log_rate.dtype, device=log_rate.device)
    if bool((elapsed < 0).any()):
        raise ValueError('Physical elapsed time must be nonnegative')
    decay = torch.exp(-log_rate.exp() * elapsed)
    angle = frequency * elapsed
    cr, ci = decay * torch.cos(angle), decay * torch.sin(angle)
    real, imaginary = cotangent[..., :n], cotangent[..., n:]
    return torch.cat((cr * real + ci * imaginary,
                      -ci * real + cr * imaginary), dim=-1)
