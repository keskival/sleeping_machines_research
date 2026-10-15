"""Multi-level addressed memory reads for persistent hierarchical composition.

Dense reference: scores nine addresses per retained lower level, per parent.
Selection is hard; selected values, persistent state and common-tape route credit
reuse ParentRaceCell. No claim of sparse candidate discovery or trained semantics.
"""
import torch
from torch import nn
from torch.nn import functional as F
from sleeping_machines.parent_race_hierarchy import ParentRaceCell


def memory_candidates(memories, level, output_shape, ancestors=True):
    """Gather content+source-level+relative-pose without pooling source memories.

memories[0] is the input surface; memories[s+1] is abstraction level s.
Addresses form a fixed 3x3 stencil at each source's native resolution, spanning
roughly the parent's footprint. This is bounded addressed access, not an
exhaustive read of every descendant. Null is always the final candidate.
"""
    oh,ow = output_shape
    blocks, masks, addresses = [], [], []
    sources = range(level+1) if ancestors else [level]
    for source in sources:
        memory = memories[source]
        b,c,h,w = memory.shape
        stride = 2**(level+1-source)
        offset = max(1,stride//2)
        content, valid, coords = [], [], []
        for dy in (-offset,0,offset):
            for dx in (-offset,0,offset):
                yy = torch.arange(oh,device=memory.device)[:,None]*stride+dy
                xx = torch.arange(ow,device=memory.device)[None,:]*stride+dx
                yy,xx = yy.expand(oh,ow),xx.expand(oh,ow)
                mask = (yy>=0)&(yy<h)&(xx>=0)&(xx<w)
                chosen = memory[:,:,yy.clamp(0,h-1),xx.clamp(0,w-1)]
                content.append(chosen.flatten(2).transpose(1,2))
                valid.append(mask.flatten()[None].expand(b,-1))
                coords.append((source,dy,dx))
        values = torch.stack(content,2)
        type_id = values.new_full((*values.shape[:-1],1),float(source)/max(1,level+1))
        poses = values.new_tensor([[dy/stride,dx/stride] for _,dy,dx in coords])
        poses = poses[None,None].expand(b,oh*ow,-1,-1)
        blocks.append(torch.cat((values,type_id,poses),-1))
        masks.append(torch.stack(valid,2)); addresses.extend(coords)
    bank = torch.cat(blocks,2); mask = torch.cat(masks,2)
    bank = torch.cat((bank,bank.new_zeros(b,oh*ow,1,c+3)),2)
    mask = torch.cat((mask,torch.ones(b,oh*ow,1,dtype=torch.bool,device=bank.device)),2)
    return bank,mask,addresses+[None]


class AddressableParentHierarchy(nn.Module):
    def __init__(self,input_width=12,width=32,levels=8,refinements=4,roles=4,classes=11,ancestors=True):
        super().__init__()
        self.width,self.levels,self.refinements,self.roles = width,levels,refinements,roles
        self.ancestors = ancestors
        self.input = nn.Conv2d(input_width,width,1)
        # The extra content coordinate identifies source abstraction level.
        self.cells = nn.ModuleList(nn.ModuleList(ParentRaceCell(width+1,width,roles)
                                   for _ in range(refinements)) for _ in range(levels))
        self.readout = nn.Linear(width,classes)

    def forward(self,features=None,memory=None,seed=0,force=None,sample=True,blocked_reads=()):
        if features is None and memory is None:
            raise ValueError('features or retained memory required')
        raw = self.input(features) if features is not None else memory['input']
        old = [None]*self.levels if memory is None else list(memory['levels'])
        if len(old)!=self.levels:
            raise ValueError('memory level count mismatch')
        generator = torch.Generator(device=raw.device).manual_seed(seed)
        memories,parents,audits = [raw],[],[]
        h,w = raw.shape[-2:]
        for level,cells in enumerate(self.cells):
            h,w = (h+1)//2,(w+1)//2
            bank,valid,addresses = memory_candidates(memories,level,(h,w),self.ancestors)
            # Preserve tape shape for controlled read-source interventions.
            blocked=[i for i,address in enumerate(addresses) if address is not None
                     and (level,address[0]) in blocked_reads]
            if blocked:
                valid=valid.clone();valid[:,:,blocked]=False
            state = raw.new_zeros(raw.shape[0],h*w,self.width) if old[level] is None else old[level].flatten(2).transpose(1,2)
            feedback = old[level+1] if level+1<self.levels else None
            if feedback is not None:
                feedback = F.interpolate(feedback,size=(h,w),mode='nearest').flatten(2).transpose(1,2)
            for refinement,cell in enumerate(cells):
                index=level*self.refinements+refinement
                override=force[1:] if force is not None and force[0]==index else None
                state,audit=cell.bind(bank,valid,state,feedback,generator,override,sample)
                audit['addresses']=addresses
                audits.append(audit)
            parent=state.transpose(1,2).reshape(raw.shape[0],self.width,h,w)
            parents.append(parent); memories.append(parent)
        return dict(logits=self.readout(parents[-1].mean((-2,-1))),audits=audits,
                    memory=dict(input=raw,levels=tuple(parents)))

    def sample_audit_site(self,features,generator):
        h,w=features.shape[-2:]; counts=[]
        for _ in range(self.levels):
            h,w=(h+1)//2,(w+1)//2
            counts.extend([h*w*self.roles]*self.refinements)
        cell=int(torch.randint(len(counts),(1,),generator=generator))
        local=int(torch.randint(counts[cell],(1,),generator=generator))
        return sum(counts[:cell])+local,1/(len(counts)*counts[cell])

    def route_objective(self,features,labels,site_index,seed=0,memory=None,site_probability=None):
        output=self(features,memory,seed)
        counts=[a['winners'].shape[1]*a['winners'].shape[2] for a in output['audits']]
        if not 0<=site_index<sum(counts):
            raise ValueError('invalid site')
        offset=site_index
        for index,count in enumerate(counts):
            if offset<count: break
            offset-=count
        audit=output['audits'][index]
        site,role=divmod(offset,self.roles)
        pi=audit['probabilities'][:,site,role]
        valid=audit['valid'][0,site]; losses=torch.zeros_like(pi)
        with torch.no_grad():
            for candidate in valid.nonzero().flatten().tolist():
                alternative=self(features,memory,seed,(index,site,role,candidate))
                losses[:,candidate]=F.cross_entropy(alternative['logits'],labels,reduction='none')
        contrast=(losses-losses[:,valid].mean(1,keepdim=True)).detach()
        q=1/sum(counts) if site_probability is None else float(site_probability)
        if not 0<q<=1: raise ValueError('invalid audit probability')
        factual=F.cross_entropy(output['logits'],labels)
        return factual+((pi-pi.detach())*contrast).sum(1).mean()/q,dict(output=output,
                    factual=factual.detach(),replays=int(valid.sum()),audit_probability=q)

    def work_descriptor(self,height,width):
        rows=[]
        for level in range(self.levels):
            height,width=(height+1)//2,(width+1)//2
            candidates=9*(level+1 if self.ancestors else 1)+1
            rows.append(dict(level=level,sites=height*width,candidates_per_site=candidates,
                      key_scores=height*width*self.roles*candidates*self.refinements,
                      selected_deliveries=height*width*self.roles*self.refinements))
        return dict(levels=rows,backend='dense gathers, all candidate values materialized',
              state='input surface plus every parent level, no implicit detach',
              credit='factual backward plus valid-candidate full replays at one audited site')


class EventAddressableParentHierarchy(AddressableParentHierarchy):
    def __init__(self,bins=8,**kwargs):
        super().__init__(input_width=12,**kwargs)
        self.bins=bins
        self.log_rates=nn.Parameter(torch.tensor([-2.,0.,2.]))
        self.frequencies=nn.Parameter(torch.tensor([0.,2.,6.]))

    def forward(self,events=None,memory=None,seed=0,force=None,sample=True,blocked_reads=()):
        from sleeping_machines.spatial_race_hierarchy import SpatialRaceHierarchy
        features=None if events is None else SpatialRaceHierarchy.surfaces(self,events)
        return super().forward(features,memory,seed,force,sample,blocked_reads)
