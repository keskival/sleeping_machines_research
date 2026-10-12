"""Same causal update schedule, separate actor/memory/readout norm budgets."""
from credit_run import run as original_run
import torch
import credit_run


def clip(model):
 groups=[[],[],[]]
 for name,p in model.named_parameters():
  index=2 if name.startswith('critic.head.') else 1 if name.startswith('critic.') else 0
  groups[index].append(p)
 for group in groups:
  if group: torch.nn.utils.clip_grad_norm_(group,1.)


def run(*args,**kwargs):
 previous=credit_run.clip;credit_run.clip=clip
 try:return original_run(*args,**kwargs)
 finally:credit_run.clip=previous
