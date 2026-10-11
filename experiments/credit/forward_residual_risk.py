"""Observation-anchored credit residual risk from reciprocal forward packets."""
import torch
@torch.no_grad()
def residual_observations(cotangent,predicted_credit,directions,forward_responses):
 if directions.shape!=forward_responses.shape or cotangent.shape!=directions.shape[1:] or predicted_credit.shape!=cotangent.shape:raise ValueError('shape mismatch')
 if not all(torch.isfinite(x).all() for x in [cotangent,predicted_credit,directions,forward_responses]):raise ValueError('nonfinite packet')
 residual=(forward_responses*cotangent).flatten(1).sum(-1)-(directions*predicted_credit).flatten(1).sum(-1)
 return residual,residual.square()
