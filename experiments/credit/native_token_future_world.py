"""R1 declared own-factual-gradient future policy for native token replay.

Not the production Bernoulli alternative-teacher policy. Native architecture,
hard races, private state, key/value maps and AdamW are retained and charged.
"""
import copy
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'experiments'))
import bernoulli_teacher_token_language as native


class NativeTokenFutureWorld:
    def __init__(self, model, optimizer, state, generator, args):
        # Joint cloning keeps optimizer parameters attached to the cloned actor.
        self.model, self.optimizer = copy.deepcopy((model, optimizer))
        self.state = native.clone_numeric_state(state)
        self.generator = torch.Generator().set_state(generator.get_state().clone())
        self.args = copy.deepcopy(args)
        self.updates = self.scored_targets = self.own_gradient_targets = 0

    def advance(self, inputs, targets, *, force_site=None):
        """Score before updating; recompute this world's gradients and moments."""
        if inputs.shape != targets.shape:
            raise ValueError('One target per observed input token')
        if self.model.force_site is not None:
            raise ValueError('Uncleared intervention on issued actor')
        self.model.train()
        self.model.force_site = force_site
        try:
            features, state, _ = self.model(inputs, self.state, self.generator, self.args)
            losses = self.model.readout.nll(features, targets).reshape_as(targets)
            if not torch.isfinite(losses).all():
                raise ValueError('Nonfinite future loss')
            self.optimizer.zero_grad(set_to_none=True)
            losses.mean().backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1., error_if_nonfinite=True)
            self.optimizer.step()
            self.state = native.detach(state)
            self.updates += 1
            self.scored_targets += targets.numel()
            self.own_gradient_targets += targets.numel()
            return losses.detach().clone()
        finally:
            self.model.force_site = None
