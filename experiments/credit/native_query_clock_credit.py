"""B10 development candidate: native target-conditioned clock credit.

Not admitted for fitting until the fixed parent confirmation finishes and
integrated null/RNG, optimizer and paired-teacher contracts pass. Receipt
construction still computes all modes; only native event processing is reduced.
"""
import copy
import torch
from native_optimizer_receipts import ReceiptCarry, events
from native_addressed_clock_credit import support


class QueryClockCredit(torch.nn.Module):
    def __init__(self, base, scale):
        super().__init__()
        self.parent = copy.deepcopy(base)
        self.parent.requires_grad_(False)
        self.receipt = copy.deepcopy(base)
        self.core = self.receipt.core
        if self.core.head.out_features != 4:
            raise ValueError('Query clock credit requires the four-leg parent')
        torch.nn.init.zeros_(self.core.head.weight)
        torch.nn.init.zeros_(self.core.head.bias)
        self.register_buffer('scale', torch.as_tensor(scale, dtype=torch.float64))
        self.last_parent = None
        self.last_mask = None

    def forward(self, features, observed, carry, operator):
        parent_carry = None if carry is None else carry.parent
        receipt_carry = None if carry is None else carry.receipt
        version = 0 if parent_carry is None else parent_carry.version
        phase = (version // 2) % 16
        self.parent.core.train(False)
        with torch.no_grad():
            baseline, parent_carry = self.parent(features, observed, parent_carry)
        # Two layer/mode queries identify the four selected layer/leg targets.
        # This first implementation still pays all-mode token preprocessing.
        tokens = events(features, observed, operator, baseline[0].detach(), self.scale)
        offset = observed.shape[1]
        indices = torch.tensor([offset + phase, offset + 16 + phase], device=tokens.device)
        queries = torch.cat((observed, tokens.index_select(1, indices)), dim=1)
        rng = torch.Generator().manual_seed(99173).get_state() if carry is None else carry.rng
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(rng)
            residual, receipt_carry = self.receipt(features, queries, receipt_carry)
            next_rng = torch.get_rng_state().clone()
        self.last_parent = baseline[0].detach().repeat_interleave(16)
        self.last_mask = support(version).to(features.device)
        selected = self.last_mask.nonzero().flatten()
        correction = residual.new_zeros(1, 64).index_copy(1, selected, .5 * residual.tanh())
        return self.last_parent[None] + correction, ReceiptCarry(parent_carry, receipt_carry, next_rng)
