"""Local clock-race supervision from observed, immutable child bindings.

Training-only candidate: it does not replace the native actor or its races.
Counterfactuals reassign existing child contents to observed poses. No class,
latent constituent identity, future observation or DEV target is accepted.
All child reads and all enumerated alternatives are paid, not called sparse.
"""
import itertools
import torch
from torch import nn
from torch.nn import functional as F


class BindingRaceTeacher(nn.Module):
    def __init__(self, width, rank=16):
        super().__init__()
        self.width, self.rank = width, rank
        self.query = nn.Linear(width, rank, bias=False)
        self.content = nn.Linear(width, rank, bias=False)
        self.pose = nn.Linear(2, rank, bias=False)

    def logits(self, parent, children, poses):
        if parent.shape != (self.width,) or children.ndim != 2 or children.shape[1] != self.width:
            raise ValueError('one parent vector and a matrix of child vectors required')
        if poses.shape != (len(children), 2) or len(children) < 2:
            raise ValueError('at least two observed child poses required')
        if not all(bool(torch.isfinite(x).all()) for x in (parent, children, poses)):
            raise ValueError('finite observed bindings required')
        # Lower views are local teaching targets. Their native whole-task credit
        # is unaffected; this auxiliary loss teaches the published parent path.
        values = F.normalize(children.detach(), dim=-1)
        locations = poses.detach() - poses.detach().mean(0)
        locations = locations / locations.square().mean().sqrt().clamp_min(1e-6)
        c, p = self.content(values), self.pose(locations)
        assignments = [tuple(range(len(children)))]
        for i, j in itertools.combinations(range(len(children)), 2):
            if torch.equal(values[i], values[j]) or torch.equal(locations[i], locations[j]):
                continue  # An indistinguishable swap has no valid target.
            order = list(assignments[0]); order[i], order[j] = order[j], order[i]
            assignments.append(tuple(order))
        keys = torch.stack([(c[list(order)] * p).sum(0) / len(children)**.5 for order in assignments])
        return keys @ self.query(F.normalize(parent, dim=0)) / self.rank**.5

    def loss(self, parent, children, poses):
        clocks = self.logits(parent, children, poses)
        if len(clocks) == 1:
            return clocks.sum()*0, dict(candidates=1, informative=False)
        # Exact first-arrival probabilities for exponential candidate clocks.
        return -F.log_softmax(clocks, dim=0)[0], dict(candidates=len(clocks), informative=True)


def observed_binding_race_loss(actor, state, teacher):
    """Teach latest published parent versions from their actual issued parts.

    Use only after draining an independent TRAIN scene. The API deliberately
    takes no scene label or generator metadata. No actor weights are updated.
    """
    if state.queue:
        raise ValueError('drain revealed events before local teaching')
    latest = {}
    for key, entry in state.bindings.entries.items():
        if key[0] < 2 or len(entry.parts) < 2:
            continue
        address = key[:2]
        if address not in latest or float(entry.available.detach()) > float(latest[address].available.detach()):
            latest[address] = entry
    losses = []; reads = candidates = children_count = 0
    for entry in latest.values():
        parent = actor.read_bound(state, entry.key, entry.available)
        children = torch.stack([actor.read_bound(state, part.source, part.issued) for part in entry.parts])
        poses = children.new_tensor([part.pose for part in entry.parts])
        loss, record = teacher.loss(parent, children, poses)
        reads += 1 + len(children); candidates += record['candidates']; children_count += len(children)
        if record['informative']: losses.append(loss)
    if not losses:
        raise ValueError('no distinguishable observed binding alternatives')
    return torch.stack(losses).mean(), dict(parents=len(latest), informative_parents=len(losses),
        bound_reads=reads, child_views=children_count, candidate_clocks=candidates,
        scope='All latest published parents and pair-swap alternatives;local teacher only. Full actor/learning arithmetic remains separately chargeable.')
