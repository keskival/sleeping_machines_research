"""Exact suffix replay for addressable persistent parent composition.

Prefix tensors are reused; downstream keys/winners are recomputed under the
same exogenous random tape. No approximation of the alternative final loss.
"""
import torch
from torch.nn import functional as F
from sleeping_machines.addressable_parent_hierarchy import AddressableParentHierarchy, memory_candidates


class CachedParentHierarchy(AddressableParentHierarchy):
    def forward(self,features=None,memory=None,seed=0,force=None,sample=True,blocked_reads=()):
        raw=self.input(features) if features is not None else memory['input']
        old=[None]*self.levels if memory is None else list(memory['levels'])
        if len(old)!=self.levels: raise ValueError('memory level count mismatch')
        generator=torch.Generator(device=raw.device).manual_seed(seed)
        memories,parents,audits,cache=[raw],[],[],[]
        h,w=raw.shape[-2:]
        for level,cells in enumerate(self.cells):
            h,w=(h+1)//2,(w+1)//2
            bank,valid,addresses=memory_candidates(memories,level,(h,w),self.ancestors)
            blocked=[i for i,address in enumerate(addresses) if address is not None and (level,address[0]) in blocked_reads]
            if blocked: valid=valid.clone();valid[:,:,blocked]=False
            state=raw.new_zeros(raw.shape[0],h*w,self.width) if old[level] is None else old[level].flatten(2).transpose(1,2)
            feedback=old[level+1] if level+1<self.levels else None
            if feedback is not None:
                feedback=F.interpolate(feedback,size=(h,w),mode='nearest').flatten(2).transpose(1,2)
            for refinement,cell in enumerate(cells):
                index=level*self.refinements+refinement
                cache.append(dict(level=level,refinement=refinement,bank=bank,valid=valid,
                    state=state,feedback=feedback,tape=generator.get_state(),memories=tuple(memories),shape=(h,w)))
                override=force[1:] if force is not None and force[0]==index else None
                state,audit=cell.bind(bank,valid,state,feedback,generator,override,sample)
                audit['addresses']=addresses;audits.append(audit)
            parent=state.transpose(1,2).reshape(raw.shape[0],self.width,h,w)
            parents.append(parent);memories.append(parent)
        return dict(logits=self.readout(parents[-1].mean((-2,-1))),audits=audits,
              memory=dict(input=raw,levels=tuple(parents)),cache=cache,old=tuple(old),
              sample=sample,blocked_reads=tuple(blocked_reads))

    def suffix_logits(self,output,index,site,role,candidate):
        """Frozen-current-parameter replay; use before optimizer changes parameters."""
        node=output['cache'][index]
        generator=torch.Generator(device=node['state'].device)
        generator.set_state(node['tape'])
        memories=list(node['memories']);state=node['state']
        h,w=node['shape'];executed=0
        for level in range(node['level'],self.levels):
            if level==node['level']:
                bank,valid=node['bank'],node['valid'];feedback=node['feedback']
                first=node['refinement']
            else:
                h,w=(h+1)//2,(w+1)//2
                bank,valid,addresses=memory_candidates(memories,level,(h,w),self.ancestors)
                blocked=[i for i,address in enumerate(addresses) if address is not None and (level,address[0]) in output['blocked_reads']]
                if blocked: valid=valid.clone();valid[:,:,blocked]=False
                previous=output['old'][level]
                state=bank.new_zeros(bank.shape[0],h*w,self.width) if previous is None else previous.flatten(2).transpose(1,2)
                feedback=output['old'][level+1] if level+1<self.levels else None
                if feedback is not None:
                    feedback=F.interpolate(feedback,size=(h,w),mode='nearest').flatten(2).transpose(1,2)
                first=0
            for refinement in range(first,self.refinements):
                override=(site,role,candidate) if executed==0 else None
                state,_=self.cells[level][refinement].bind(bank,valid,state,feedback,generator,override,output['sample'])
                executed+=1
            parent=state.transpose(1,2).reshape(state.shape[0],self.width,h,w)
            memories.append(parent)
        return self.readout(parent.mean((-2,-1))),executed

    def route_objective(self,features,labels,site_index,seed=0,memory=None,site_probability=None):
        output=self(features,memory,seed)
        counts=[a['winners'].shape[1]*a['winners'].shape[2] for a in output['audits']]
        if not 0<=site_index<sum(counts): raise ValueError('invalid site')
        offset=site_index
        for index,count in enumerate(counts):
            if offset<count: break
            offset-=count
        audit=output['audits'][index];site,role=divmod(offset,self.roles)
        pi=audit['probabilities'][:,site,role];valid=audit['valid'][0,site]
        losses=torch.zeros_like(pi);executed=0
        with torch.no_grad():
            for candidate in valid.nonzero().flatten().tolist():
                logits,work=self.suffix_logits(output,index,site,role,candidate)
                executed+=work
                losses[:,candidate]=F.cross_entropy(logits,labels,reduction='none')
        contrast=(losses-losses[:,valid].mean(1,keepdim=True)).detach()
        q=1/sum(counts) if site_probability is None else float(site_probability)
        if not 0<q<=1: raise ValueError('invalid audit probability')
        factual=F.cross_entropy(output['logits'],labels)
        return factual+((pi-pi.detach())*contrast).sum(1).mean()/q,dict(output=output,
              factual=factual.detach(),replays=int(valid.sum()),audit_probability=q,
              audited_cell=index,suffix_cell_evaluations=executed,
              full_replay_cell_evaluations=int(valid.sum())*self.levels*self.refinements)


class EventCachedParentHierarchy(CachedParentHierarchy):
    def __init__(self,bins=8,**kwargs):
        super().__init__(input_width=12,**kwargs)
        self.bins=bins
        self.log_rates=torch.nn.Parameter(torch.tensor([-2.,0.,2.]))
        self.frequencies=torch.nn.Parameter(torch.tensor([0.,2.,6.]))

    def forward(self,events=None,memory=None,seed=0,force=None,sample=True,blocked_reads=()):
        from sleeping_machines.spatial_race_hierarchy import SpatialRaceHierarchy
        features=None if events is None else SpatialRaceHierarchy.surfaces(self,events)
        return super().forward(features,memory,seed,force,sample,blocked_reads)
