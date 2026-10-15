"""Final-task counterfactual credit for the spatial race hierarchy.

Reference implementation: full replays, all candidate proposals materialized.
No claim of sparse execution or measured depth-scaling improvement.
"""
import torch
from torch import nn
from torch.nn import functional as F
from sleeping_machines.spatial_race_hierarchy import FeatureRaceBlock, SpatialRaceHierarchy


class ReplayRaceBlock(FeatureRaceBlock):
    def forward(self, x, generator=None, sample=None):
        b, c, h, w = x.shape
        u = F.gelu(self.compose(self.norm(x)))
        scores = self.keys(u).view(b, self.groups, self.candidates, h, w)
        pi = scores.softmax(2)
        sample = self.training if sample is None else sample
        if sample:
            uniform = torch.rand(scores.shape, dtype=scores.dtype, device=scores.device,
                                 generator=generator).clamp(torch.finfo(scores.dtype).tiny, 1-torch.finfo(scores.dtype).eps)
            winners = (torch.log(-torch.log(uniform))-scores).argmin(2)
        else:
            winners = scores.argmax(2)
        force = getattr(self, '_force', None)
        if force is not None:
            group, yy, xx, candidate = force
            winners = winners.clone()
            winners[:, group, yy, xx] = candidate
        values = self.values(u).view(b, self.groups, self.candidates, c//self.groups, h, w)
        chosen = values.gather(2, winners[:, :, None, None].expand(-1, -1, 1, c//self.groups, -1, -1)).squeeze(2)
        out = x + self.scale*self.mix(chosen.reshape(b, c, h, w))
        return out, dict(probabilities=pi, values=values, chosen=chosen,
                         winners=winners, scale=self.scale, mix=self.mix)


class RelationalComposition(nn.Module):
    """Shared child-role products supplement an arrangement-sensitive linear path.

Four relative child positions remain distinct. Low-rank pair products encode
conjunctions: the same child features can participate in different parents.
No semantic part labels or object-binding guarantee is assumed.
"""
    def __init__(self, child_width, parent_width):
        super().__init__()
        self.linear = nn.Conv2d(child_width, parent_width, 3, stride=2, padding=1)
        self.roles = nn.ModuleList(nn.Conv2d(child_width, parent_width, 1)
                                   for _ in range(4))
        self.relations = nn.Conv2d(6*parent_width, parent_width, 1, bias=False)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        # Match stride-2 3x3 output shape, including odd sizes.
        padded = F.pad(x, (0, x.shape[-1] % 2, 0, x.shape[-2] % 2))
        children = [padded[:, :, dy::2, dx::2] for dy, dx in
                    ((0,0), (0,1), (1,0), (1,1))]
        roles = [torch.tanh(proj(child)) for proj, child in zip(self.roles, children)]
        pairs = torch.cat([roles[i]*roles[j] for i in range(4) for j in range(i+1,4)], 1)
        return self.linear(x)+self.scale*self.relations(pairs)


class DownstreamRaceHierarchy(SpatialRaceHierarchy):
    """Same representational graph; replaces local class credit with final loss.

A single uniformly sampled route site is audited per update. All K alternative
suffixes share the factual random tape. This gives unbiased gate credit over
site sampling and race noise, but can have high importance-weight variance.
Continuous parameters receive factual pathwise credit. Intermediate heads are
readouts only; no same-label auxiliary objective is imposed on early features.
"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for level in range(1, len(self.widths)):
            self.transitions[level] = RelationalComposition(self.widths[level-1], self.widths[level])
        for stage in self.stages:
            for i, old in enumerate(stage):
                new = ReplayRaceBlock(old.width, old.groups, old.candidates,
                                      float(old.scale.detach()))
                new.load_state_dict(old.state_dict())
                stage[i] = new

    def replay(self, events, seed, force=None):
        blocks = [block for stage in self.stages for block in stage]
        if force is not None:
            index, group, yy, xx, candidate = force
            blocks[index]._force = (group, yy, xx, candidate)
        generator = torch.Generator(device=events.device).manual_seed(seed)
        try:
            return self(events, generator=generator, sample=True)
        finally:
            if force is not None:
                del blocks[index]._force

    def downstream_objective(self, events, labels, seed, site_index):
        """site_index must be drawn uniformly independently of the factual tape.

The returned zero-forward gate surrogate has the derivative of the exact
conditional alternative risk at the audited site, divided by site probability.
No clipping/normalization: those would change the declared estimator.
"""
        output = self.replay(events, seed)
        counts = [a['winners'].shape[1]*a['winners'].shape[2]*a['winners'].shape[3]
                  for a in output['audits']]
        total = sum(counts)
        if not 0 <= site_index < total:
            raise ValueError('site_index outside route-site population')
        offset = site_index
        for index, count in enumerate(counts):
            if offset < count:
                break
            offset -= count
        audit = output['audits'][index]
        _, _, h, w = audit['winners'].shape
        group, spatial = divmod(offset, h*w)
        yy, xx = divmod(spatial, w)
        losses = []
        with torch.no_grad():
            for candidate in range(self.candidates):
                alternative = self.replay(events, seed, (index, group, yy, xx, candidate))
                losses.append(F.cross_entropy(alternative['logits'], labels, reduction='none'))
        losses = torch.stack(losses, 1)
        # Subtracting a candidate-independent baseline leaves the exact derivative.
        losses = losses-losses.mean(1, keepdim=True)
        pi = audit['probabilities'][:, group, :, yy, xx]
        gate = total*((pi-pi.detach())*losses).sum(1).mean()
        factual = F.cross_entropy(output['logits'], labels)
        return factual+gate, dict(factual=factual.detach(), site_count=total,
                                 audited_block=index, counterfactual_replays=self.candidates,
                                 alternative_loss_range=(losses.max()-losses.min()).detach())
