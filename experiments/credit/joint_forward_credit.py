"""Train a joint credit estimator from detached, measured forward responses.

Only estimator predictions receive gradients. Probe/actor/terminal-credit
versions are fixed for this observation. Independent isotropic cut probes give
the joint credit-regression gradient in expectation; shared producer probes
give its producer-projected gradient, including cross terms. Rank-deficient
probe support does not identify credit outside that support.
"""
import torch


def joint_credit_teaching_loss(predicted_content, predicted_time, cotangent,
                               content_directions, time_directions, responses):
    if (content_directions.shape != responses.shape or
            cotangent.shape != responses.shape[1:] or
            predicted_content.shape != content_directions.shape[1:] or
            time_directions.shape != content_directions.shape[:-1] or
            predicted_time.shape != time_directions.shape[1:] or
            len(responses) == 0):
        raise ValueError('joint teaching shape mismatch')
    if not all(torch.isfinite(x).all() for x in [predicted_content, predicted_time,
               cotangent, content_directions, time_directions, responses]):
        raise ValueError('nonfinite joint teaching observation')
    measured = (responses.detach() * cotangent.detach()).flatten(1).sum(-1)
    predicted = ((content_directions.detach() * predicted_content).flatten(1).sum(-1) +
                 (time_directions.detach() * predicted_time).flatten(1).sum(-1))
    return .5 * (predicted - measured).square().mean()
