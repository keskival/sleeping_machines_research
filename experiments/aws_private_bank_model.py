"""Shared processing rules, replicated private addresses, unchanged event kernel.

Construction always uses the same canonical pool, so readout/shared maps and
post-construction RNG match. Replicas start equal, then acquire private state.
"""
import copy
import hashlib
from pathlib import Path
import torch
from torch import nn
import horizon_token_language_engine as lab
from aws_private_bank_policy import replica_policy


class PrivateBankModel(lab.Model):
    def __init__(self, args, order, counts, canonical_pool=4):
        if not args.tie_pools or args.temperature != 1. or args.free_bias != 0.:
            raise ValueError('Bank comparison requires shared rules, temperature one, zero free bias')
        self.bank_policy = replica_policy(canonical_pool, args.pool)
        canonical_args = copy.copy(args)
        canonical_args.pool = canonical_pool
        super().__init__(canonical_args, order, counts)
        replicas = self.bank_policy['replicas']
        for name in self.bank_policy['shared_fields']:
            if not self.core.tied[name]: raise ValueError('Processing rule was not shared: '+name)
        for name in self.bank_policy['private_fields']:
            if self.core.tied[name]: raise ValueError('Private bank unexpectedly shared: '+name)
            value = self.core.banks[name].detach().repeat_interleave(replicas, dim=2)
            if name == 'clock_bias': value = value + self.bank_policy['clock_bias_shift']
            self.core.banks[name] = nn.Parameter(value.clone())
        self.core.pool = args.pool
        self.bank_source_sha256 = {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ('aws_private_bank_model.py','aws_private_bank_policy.py')}

    def get_extra_state(self):
        return dict(bank_policy=self.bank_policy, source_sha256=self.bank_source_sha256)

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError('Private-bank checkpoint recipe/source mismatch')
