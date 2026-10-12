"""Native critic memory taught by forward task loss, auxiliary readout locally.
Diagnosed response to600--1200x regression/task gradient imbalance.
"""
from torch import nn
from torch.nn import functional as F
from shared_bank_sensor_v2 import SharedBankSensor


class LocalLossReadout(nn.Module):
 def __init__(self, linear):
  super().__init__(); self.weight=linear.weight; self.bias=linear.bias
 def forward(self, x): return F.linear(x.detach(), self.weight, self.bias)


class TaskTaughtSharedBank(SharedBankSensor):
 def __init__(self, confidence, mode='reciprocal'):
  super().__init__(confidence, mode)
  # Identical forward function/parameters/RNG, distinct teaching path.
  self.critic.head=LocalLossReadout(self.critic.head)
