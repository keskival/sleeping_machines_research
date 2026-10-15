"""Persistent role-binding hierarchy: development reference, dense execution.

Depth advances representation scale; refinement steps within a scale are separate.
State is explicit and is never implicitly detached or reset. Hard categorical
choices require final-task route credit; ordinary backprop alone is incomplete.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class ParentRaceCell(nn.Module):
    def __init__(self, child_width, width, roles=4, radius=1, transport=True):
        super().__init__()
        self.width, self.roles, self.radius = width, roles, radius
        self.use_transport = transport
        self.key = nn.Linear(child_width+2, width, bias=False)
        self.value = nn.Linear(child_width+2, width)
        self.query = nn.Linear(width, roles*width)
        self.role_queries = nn.Parameter(torch.randn(roles, width)/math.sqrt(width))
        self.relate_left = nn.Linear(width, width, bias=False)
        self.relate_right = nn.Linear(width, width, bias=False)
        pairs = roles*(roles-1)//2
        self.proposal = nn.Linear((roles+pairs)*width, width)
        self.write_gate = nn.Linear(width, width)
        nn.init.constant_(self.write_gate.bias, 0. if transport else -2.)
        self.feedback = nn.Linear(width, width, bias=False)
        self.transport = nn.Linear(child_width, width, bias=False)
        with torch.no_grad():
            self.transport.weight.zero_()
            self.transport.weight[:min(child_width,width),:min(child_width,width)] = torch.eye(min(child_width,width))

    def candidates(self, children):
        """Local addressed children with explicit relative pose, plus null."""
        b, c, h, w = children.shape
        kernel = 2*self.radius+1
        patch = F.unfold(children, kernel, padding=self.radius, stride=2)
        oh, ow = (h+1)//2, (w+1)//2
        patch = patch.view(b,c,kernel*kernel,oh*ow).permute(0,3,2,1)
        yy, xx = torch.meshgrid(torch.arange(-self.radius,self.radius+1,device=children.device),
                                torch.arange(-self.radius,self.radius+1,device=children.device), indexing='ij')
        pose = torch.stack((yy,xx),-1).reshape(1,1,-1,2).to(children.dtype).expand(b,oh*ow,-1,-1)
        candidates = torch.cat((patch,pose),-1)
        valid = F.unfold(torch.ones(b,1,h,w,device=children.device,dtype=children.dtype),
                         kernel,padding=self.radius,stride=2).transpose(1,2).bool()
        # Null is a true zero value, not an invented observed child.
        candidates = torch.cat((candidates, candidates.new_zeros(b,oh*ow,1,c+2)),2)
        valid = torch.cat((valid, torch.ones(b,oh*ow,1,device=children.device,dtype=torch.bool)),2)
        return candidates, valid, (oh,ow)

    def bind(self, candidates, valid, state, feedback=None, generator=None, force=None, sample=True):
        b, n, k, _ = candidates.shape
        context = state if feedback is None else state+self.feedback(feedback)
        query = self.query(context).view(b,n,self.roles,self.width)+self.role_queries
        keys = self.key(candidates)
        scores = torch.einsum('bnrd,bnkd->bnrk',query,keys)/math.sqrt(self.width)
        scores = scores.masked_fill(~valid[:,:,None], -torch.inf)
        pi = scores.softmax(-1)
        if sample:
            uniform = torch.rand(scores.shape,device=scores.device,dtype=scores.dtype,generator=generator)
            uniform = uniform.clamp(torch.finfo(scores.dtype).tiny,1-torch.finfo(scores.dtype).eps)
            winners = (torch.log(-torch.log(uniform))-scores).argmin(-1)
        else:
            winners = scores.argmax(-1)
        if force is not None:
            site, role, candidate = force
            if not valid[:,site,candidate].all():
                raise ValueError('cannot force a padding candidate')
            winners = winners.clone(); winners[:,site,role] = candidate
        values = self.value(candidates)
        if self.use_transport:
            values = self.transport(candidates[...,:-2])+0.1*values
        # Null never introduces content through projection bias.
        values = torch.cat((values[:,:,:-1],torch.zeros_like(values[:,:,-1:])),2)
        chosen = values.gather(2,winners[...,None].expand(-1,-1,-1,self.width))
        left, right = torch.tanh(self.relate_left(chosen)), torch.tanh(self.relate_right(chosen))
        relations = [left[:,:,i]*right[:,:,j] for i in range(self.roles) for j in range(i+1,self.roles)]
        joint = torch.cat([chosen.flatten(2), *relations],-1)
        proposal = torch.tanh(self.proposal(joint))
        if self.use_transport:
            # Preserve selected child content while adding relational refinement.
            # Interpolation avoids adding the same evidence unboundedly on refinement.
            proposal = chosen.mean(2)+0.1*proposal-state
        delta = torch.sigmoid(self.write_gate(context))*proposal
        active = (winners != k-1).any(-1,keepdim=True)
        updated = state+active*delta
        return updated, dict(probabilities=pi,winners=winners,valid=valid,chosen=chosen,
                             emission=updated-state, active=active)

    def forward(self, children, state=None, feedback=None, generator=None, force=None, sample=True):
        candidates, valid, (h,w) = self.candidates(children)
        b = children.shape[0]
        old = children.new_zeros(b,h*w,self.width) if state is None else state.flatten(2).transpose(1,2)
        fb = None if feedback is None else F.interpolate(feedback,size=(h,w),mode='nearest').flatten(2).transpose(1,2)
        new, audit = self.bind(candidates,valid,old,fb,generator,force,sample)
        return new.transpose(1,2).reshape(b,self.width,h,w), audit


class ParentRaceHierarchy(nn.Module):
    """Eight abstraction levels x four refinements = 32 learned parent cells.

Input is an event feature surface [B,input_width,H,W], not a longer sequence.
Optional old state persists across calls. Upper old states query the level below;
new lower states flow upward in the current call. No cyclic implicit solver.
"""
    def __init__(self, input_width=12, width=32, levels=8, refinements=4, roles=4, classes=11, transport=True):
        super().__init__()
        self.levels, self.refinements, self.width = levels, refinements, width
        self.cells = nn.ModuleList(nn.ModuleList(ParentRaceCell(input_width if level==0 else width,
                          width,roles,transport=transport) for _ in range(refinements)) for level in range(levels))
        self.readout = nn.Linear(width,classes)

    def forward(self, features, state=None, seed=0, force=None, sample=True):
        if state is not None and len(state)!=self.levels:
            raise ValueError('state level count mismatch')
        generator = torch.Generator(device=features.device).manual_seed(seed)
        old_states = [None]*self.levels if state is None else list(state)
        new_states, audits = [], []
        children = features
        for level, cells in enumerate(self.cells):
            parent = old_states[level]
            feedback = old_states[level+1] if level+1<self.levels else None
            for refinement, cell in enumerate(cells):
                index = level*self.refinements+refinement
                override = force[1:] if force is not None and force[0]==index else None
                parent, audit = cell(children,parent,feedback,generator,override,sample)
                audits.append(audit)
            new_states.append(parent)
            children = parent
        return dict(logits=self.readout(children.mean((-2,-1))),state=new_states,audits=audits)

    def sample_audit_site(self, features, generator):
        """Balance cells, then sites within cell; use a separate audit RNG."""
        h,w = features.shape[-2:]
        counts = []
        for level,cells in enumerate(self.cells):
            h,w = (h+1)//2,(w+1)//2
            counts.extend(h*w*cell.roles for cell in cells)
        cell = int(torch.randint(len(counts),(1,),generator=generator))
        local = int(torch.randint(counts[cell],(1,),generator=generator))
        return sum(counts[:cell])+local, 1/(len(counts)*counts[cell])

    def route_objective(self, features, labels, site_index, seed=0, state=None, site_probability=None):
        """Uniform independent site audit; enumerate valid children through final loss.

Population includes each cell/site/role. Null is included; padding is excluded
from candidate probabilities and alternative replays. Replays reuse old state
without mutation; factual selected-value gradients and categorical terms add.
"""
        output = self(features,state,seed)
        counts = [a['winners'].shape[1]*a['winners'].shape[2] for a in output['audits']]
        total = sum(counts)
        if not 0 <= site_index < total:
            raise ValueError('site index outside population')
        offset = site_index
        for index,count in enumerate(counts):
            if offset<count:
                break
            offset-=count
        audit = output['audits'][index]
        site,role = divmod(offset,audit['winners'].shape[2])
        pi = audit['probabilities'][:,site,role]
        # Spatial validity is identical across batch elements by construction.
        valid = audit['valid'][0,site]
        losses = pi.new_zeros(pi.shape)
        with torch.no_grad():
            for candidate in valid.nonzero().flatten().tolist():
                replay = self(features,state,seed,(index,site,role,candidate))
                losses[:,candidate] = F.cross_entropy(replay['logits'],labels,reduction='none')
        baseline = losses[:,valid].mean(1,keepdim=True)
        contrast = (losses-baseline).detach()
        probability = 1/total if site_probability is None else float(site_probability)
        if not 0 < probability <= 1:
            raise ValueError('invalid audit probability')
        gate = ((pi-pi.detach())*contrast).sum(1).mean()/probability
        factual = F.cross_entropy(output['logits'],labels)
        return factual+gate, dict(factual=factual.detach(),site_count=total,
                  audited_cell=index,audit_probability=probability,replays=int(valid.sum()),output=output)


class EventParentRaceHierarchy(ParentRaceHierarchy):
    """Fixed-window event surfaces feeding the persistent parent construction."""
    def __init__(self, bins=8, **kwargs):
        super().__init__(input_width=12, **kwargs)
        self.bins = bins
        self.log_rates = nn.Parameter(torch.tensor([-2.,0.,2.]))
        self.frequencies = nn.Parameter(torch.tensor([0.,2.,6.]))

    def forward(self, events, state=None, seed=0, force=None, sample=True):
        # Reuse the established learned decay/rotation adapter exactly.
        from sleeping_machines.spatial_race_hierarchy import SpatialRaceHierarchy
        features = SpatialRaceHierarchy.surfaces(self,events)
        return super().forward(features,state,seed,force,sample)
