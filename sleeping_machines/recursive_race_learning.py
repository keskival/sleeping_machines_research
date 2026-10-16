"""Local derived-stream learning with the existing native competing-risk law.

One forecast is frozen until its target arrives or it is explicitly replaced.
Queries split exposure; they never reset clocks or create training events.
This adapter does not choose concept identities or emission boundaries.
"""
from dataclasses import dataclass
import torch
from torch import nn
from .race_readout import RaceReadout, log_surv, event_terms


@dataclass
class RaceExposure:
    laws: dict
    references: torch.Tensor
    cursor: torch.Tensor
    eps: float
    closed: bool = False

    def _endpoint(self, time):
        t = torch.as_tensor(time, dtype=self.cursor.dtype, device=self.cursor.device)
        if self.closed or t.shape != self.cursor.shape or not bool(torch.isfinite(t).all()) or bool((t < self.cursor).any()):
            raise ValueError('open forecast and finite nondecreasing per-lane times required')
        return t

    def silence(self, time):
        """Negative log conditional survival ONLY on (cursor,time]."""
        t = self._endpoint(time)
        old = log_surv(self.laws, self.cursor[:, None]-self.references, self.eps)
        new = log_surv(self.laws, t[:, None]-self.references, self.eps)
        loss = (old-new).sum(-1)
        self.cursor = t.clone()
        return loss

    def event(self, time, marks):
        """Point-density NLL; the mark is an observed, nondifferentiable label.

        Caller provides availability times and a non-atomic continuous clock.
        Quantized/tied streams need a separate exact recording-cell law.
        """
        t = self._endpoint(time)
        labels = torch.as_tensor(marks, device=t.device)
        if labels.shape != t.shape or labels.dtype != torch.long:
            raise ValueError('one integer identity label per forecast lane required')
        if bool(((labels < 0) | (labels >= self.laws['logp'].shape[-1])).any()):
            raise ValueError('identity outside this level vocabulary')
        total, timing, posterior = event_terms(self.laws, self.references, self.cursor, t, labels.detach(), self.eps)
        self.cursor = t.clone(); self.closed = True
        return -total, -timing, posterior


class NativeRecursiveRaceHeads(nn.Module):
    """Training-only native read adapter; retains published object values/times.

    Pools are explicitly supplied, bounded, and charged. No private future
    state, hidden part ids, actor mutation, or symbolic tree is used.
    Selection/discovery costs outside the supplied pool remain caller costs.
    """
    def __init__(self, width, levels, slots=4, identities=16, hidden=32, eps=1e-3):
        super().__init__()
        self.width, self.slots, self.eps = width, slots, eps
        self.heads = nn.ModuleList(RaceReadout(identities, 1, slots, width, width,
                                  hidden=hidden, context='additive') for _ in range(levels))

    def forecast(self, actor, state, level, addresses):
        if not 0 <= level < len(self.heads) or len(addresses) != self.slots or len(set(addresses)) != self.slots:
            raise ValueError('one distinct published address per declared pool slot required')
        obj, bank = actor.objects[level], state.banks[level]
        values, times = [], []
        for address in addresses:
            if address not in bank.published: raise ValueError('only published memories are eligible')
            value, _ = obj.read(bank, address, state.now)
            record = bank.published[address]
            values.append(value)
            times.append(torch.as_tensor(record.ready, dtype=torch.float64, device=value.device))
        memory = torch.stack(values)[None, None]
        laws = self.heads[level](memory, memory.mean((1,2)))
        refs = torch.stack(times)[None]
        now = refs.new_tensor([state.now])
        if bool((refs > now[:,None]).any()): raise ValueError('future publication in forecast')
        # Freeze tensors by functional snapshot: later actor transitions cannot
        # change these laws. Keep gradients for local representation credit.
        ledger = RaceExposure(laws, refs, now, self.eps)
        work = dict(published_reads=self.slots, scored_race_slots=self.slots,
                    mark_logits=self.slots*self.heads[level].types,
                    retained_law_scalars=sum(x.numel() for x in laws.values()),
                    readout_madds=self.slots*(self.width*self.heads[level].mem_in.out_features+
                        self.heads[level].out.in_features*self.heads[level].out.out_features)+
                        self.width*self.heads[level].out.out_features,
                    boundary='Partial forward ledger; transport, nonlinear/special functions, backward, optimizer, address discovery and actor work separate.')
        return ledger, work
