"""Forward-observed residual labels for joint content/time credit.

Use the same probe for both outputs of a shared producer. Independent output
probes estimate cut risk, but cannot replace shared parameter probes when
measuring the producer update's error.
"""
import torch


@torch.no_grad()
def joint_residual_observations(cotangent, predicted_content, predicted_time,
                                content_directions, time_directions, responses):
    if (content_directions.shape != responses.shape or
            cotangent.shape != responses.shape[1:] or
            predicted_content.shape != content_directions.shape[1:] or
            time_directions.shape != content_directions.shape[:-1] or
            predicted_time.shape != time_directions.shape[1:]):
        raise ValueError('joint risk shape mismatch')
    if not all(torch.isfinite(x).all() for x in [cotangent, predicted_content,
               predicted_time, content_directions, time_directions, responses]):
        raise ValueError('nonfinite joint risk packet')
    residual = ((responses * cotangent).flatten(1).sum(-1) -
                (content_directions * predicted_content).flatten(1).sum(-1) -
                (time_directions * predicted_time).flatten(1).sum(-1))
    return residual, residual.square()
