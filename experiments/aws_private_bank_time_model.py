"""Private-bank model with a positive, source-bound memory time convention.

Every factual and forced alternative forward uses the same scale. Gradients
flow through the scaled coefficients. No stored parameter is mutated.
"""
import hashlib
import math
from pathlib import Path

from aws_private_bank_model import PrivateBankModel
from aws_memory_time_scale import installed_time_scale
import importlib


class TimeScaledPrivateBankModel(PrivateBankModel):
    def __init__(self, args, order, counts, memory_time_scale=1.):
        if not math.isfinite(memory_time_scale) or memory_time_scale <= 0:
            raise ValueError('Finite positive memory time scale required')
        if args.compiled:
            raise ValueError('Time-scale model requires the audited eager kernel')
        self.memory_time_scale = memory_time_scale
        super().__init__(args, order, counts)

    def forward(self, *args, **kwargs):
        kernel = importlib.import_module('sleeping_machines.sparse_counterfactual_layer')
        with installed_time_scale(kernel, self.memory_time_scale):
            return super().forward(*args, **kwargs)

    def get_extra_state(self):
        state = super().get_extra_state()
        state['memory_time_scale'] = self.memory_time_scale
        state['time_source_sha256'] = {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ('aws_memory_time_scale.py', 'aws_private_bank_time_model.py')}
        return state
