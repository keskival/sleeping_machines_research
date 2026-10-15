"""Deep spatial representation composition, independent of sequence length.

Own-family feature races: score keys select hard candidate values; per-stage
conditional readout audits teach losing candidate values and route keys. The
current tensor backend evaluates all candidate proposals and spatial sites.
This is not a claim of event-sparse convolution or full future-write credit.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class ChannelNorm(nn.Module):
    """Normalize channels at each site; never mix spatial sites through statistics."""
    def __init__(self, width):
        super().__init__(); self.norm = nn.LayerNorm(width)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class FeatureRaceBlock(nn.Module):
    def __init__(self, width, groups=2, candidates=4, residual_scale=0.1):
        super().__init__()
        if width % groups or candidates < 2:
            raise ValueError('width must divide into groups; at least two candidates')
        self.width, self.groups, self.candidates = width, groups, candidates
        self.norm = ChannelNorm(width)
        # A local arrangement-sensitive composition, then separate selection/value paths.
        self.compose = nn.Conv2d(width, width, 3, padding=1)
        self.keys = nn.Conv2d(width, groups*candidates, 1)
        self.values = nn.Conv2d(width, width*candidates, 1)
        self.mix = nn.Conv2d(width, width, 1, bias=False)
        self.scale = nn.Parameter(torch.tensor(float(residual_scale)))

    def forward(self, x, generator=None, sample=None):
        b, c, h, w = x.shape
        u = F.gelu(self.compose(self.norm(x)))
        scores = self.keys(u).view(b, self.groups, self.candidates, h, w)
        probabilities = scores.softmax(2)
        sample = self.training if sample is None else sample
        if sample:
            uniform = torch.rand(scores.shape, dtype=scores.dtype, device=scores.device,
                                 generator=generator).clamp_min(torch.finfo(scores.dtype).tiny)
            # log exponential-race arrival: log(-log U) - score.
            winners = (torch.log(-torch.log(uniform))-scores).argmin(2)
        else:
            winners = scores.argmax(2)
        values = self.values(u).view(b, self.groups, self.candidates, c//self.groups, h, w)
        chosen = values.gather(2, winners[:, :, None, None].expand(-1, -1, 1, c//self.groups, -1, -1)).squeeze(2)
        selected = chosen.reshape(b, c, h, w)
        out = x + self.scale*self.mix(selected)
        return out, dict(probabilities=probabilities, values=values, chosen=chosen,
                         winners=winners, scale=self.scale, mix=self.mix)


class SpatialRaceHierarchy(nn.Module):
    """Default 32 representation blocks / 8 scales, not 32 time steps.

Input [batch,time_bin,polarity,y,x] holds the same observed evidence at every
model depth. Event-time weights form learned decaying/rotating surfaces. All
subsequent depth composes spatial features into increasingly broad context.
"""
    def __init__(self, classes=11, bins=8, blocks_per_stage=4,
                 widths=(16, 24, 32, 48, 64, 80, 96, 128), groups=2, candidates=4):
        super().__init__()
        if len(widths) < 3 or blocks_per_stage < 1:
            raise ValueError('Need >=3 spatial stages and >=1 block per stage')
        self.bins, self.widths = bins, tuple(widths)
        self.blocks_per_stage, self.groups, self.candidates = blocks_per_stage, groups, candidates
        self.log_rates = nn.Parameter(torch.tensor([-2.0, 0.0, 2.0]))
        self.frequencies = nn.Parameter(torch.tensor([0.0, 2.0, 6.0]))
        self.stem = nn.Conv2d(12, widths[0], 3, padding=1)
        self.transitions = nn.ModuleList()
        self.stages = nn.ModuleList()
        self.heads = nn.ModuleList()
        for level, width in enumerate(widths):
            self.transitions.append(nn.Identity() if level == 0 else
                                    nn.Conv2d(widths[level-1], width, 3, stride=2, padding=1))
            self.stages.append(nn.ModuleList(FeatureRaceBlock(width, groups, candidates,
                                    1/math.sqrt(2*len(widths)*blocks_per_stage))
                                    for _ in range(blocks_per_stage)))
            self.heads.append(nn.Linear(width, classes))

    def surfaces(self, events):
        if events.ndim != 5 or events.shape[1:3] != (self.bins, 2):
            raise ValueError('Expected [B,bins,2,H,W] event counts')
        # Time is fixed within [0,1]; bins represent the same prefix for every depth.
        age = 1-(torch.arange(self.bins, device=events.device, dtype=events.dtype)+0.5)/self.bins
        mag = torch.exp(-self.log_rates.exp()[:, None]*age[None])
        angle = self.frequencies[:, None]*age[None]
        kernel = torch.cat([mag*angle.cos(), mag*angle.sin()], 0)
        # Log counts bound dynamic range; this is an explicit nonlinear input adapter.
        return torch.einsum('mt,btpyx->bmpyx', kernel, events.log1p()).flatten(1, 2)

    def forward(self, events, generator=None, sample=None, ablate_stage=None):
        x = F.gelu(self.stem(self.surfaces(events)))
        logits, audits, features, block_features, head_indices = [], [], [], [], []
        for level, (transition, stage, head) in enumerate(zip(self.transitions, self.stages, self.heads)):
            x = transition(x)
            last = None
            for block in stage:
                x, last = block(x, generator=generator, sample=sample)
                audits.append(last); block_features.append(x); head_indices.append(level)
            if ablate_stage == level:
                # Diagnostic only: removes spatial distinctions, retains per-channel mean.
                x = x.mean((-2, -1), keepdim=True).expand_as(x)
            logits.append(head(x.mean((-2, -1))))
            features.append(x)
        return dict(logits=logits[-1], stage_logits=logits, audits=audits, features=features, block_features=block_features, head_indices=head_indices)

    def objective(self, output, labels, auxiliary=0.2, counterfactual=0.1, audit_site=0):
        """Global factual BP plus direct stage supervision and exact local race audit.

At every block, replace one spatial site's one group at a time
with every alternative. The immediate pooled linear head's counterfactual loss
is exact. Detached other sites fix the conditioning; both probabilities and
alternative values are trained, holding the current mixer/readout fixed.
        This is not full downstream expected-risk
credit. Cycling sites offers coverage without enumerating every image site.
"""
        main = F.cross_entropy(output['logits'], labels)
        aux = torch.stack([F.cross_entropy(lg, labels) for lg in output['stage_logits'][:-1]]).mean()
        cf = main.new_zeros(())
        for feature, audit, level in zip(output['block_features'], output['audits'], output['head_indices']):
            head = self.heads[level]
            b, c, h, w = feature.shape
            site = audit_site % (h*w); yy, xx = divmod(site, w)
            base = feature.mean((-2, -1)).detach()
            values = audit['values'][:, :, :, :, yy, xx]
            selected = audit['chosen'][:, :, :, yy, xx].detach()
            pi = audit['probabilities'][:, :, :, yy, xx]
            # Mixing is a 1x1 linear map. Per-group alternatives have exact pooled effects.
            mix = audit['mix'].weight[:, :, 0, 0].detach()
            for g in range(self.groups):
                delta = values[:, g]-selected[:, g, None]
                mapped = F.linear(delta, mix[:, g*(c//self.groups):(g+1)*(c//self.groups)])
                alternative = base[:, None] + audit['scale'].detach()*mapped/(h*w)
                lg = F.linear(alternative, head.weight.detach(), head.bias.detach())
                losses = F.cross_entropy(lg.reshape(-1, lg.shape[-1]),
                           labels[:, None].expand(-1, self.candidates).reshape(-1), reduction='none').view(b, -1)
                cf = cf + (pi*losses).sum(-1).mean()/(len(output['audits'])*self.groups)
        return main + auxiliary*aux + counterfactual*cf, dict(main=main.detach(), auxiliary=aux.detach(),
                                                            local_counterfactual=cf.detach())

    def work_descriptor(self, height=128, width=128):
        sizes = []
        for i, c in enumerate(self.widths):
            if i:
                height, width = (height+1)//2, (width+1)//2
            sizes.append(dict(height=height, width=width, channels=c, blocks=self.blocks_per_stage,
                              selected_routes=height*width*self.groups*self.blocks_per_stage,
                              scored_routes=height*width*self.groups*self.candidates*self.blocks_per_stage))
        return dict(representation_blocks=len(self.widths)*self.blocks_per_stage, scales=sizes,
                    backend='dense spatial tensors; all proposals evaluated',
                    sequence_depth='fixed observed bins; not increased with representation depth',
                    missing='event-sparse dispatch, cross-window carry, full downstream alternative-write credit')
