"""Native carried hierarchy with explicitly charged multiscale output reads.

All lower memories remain intact. This stage learns evidence allocation; it does
not suppress transmissions or claim a sparse output selection implementation.
"""
import torch
from .carried_role_composition import CarriedRoleComposition
from .responsibility_readout import ResponsibilityReadout


class ResponsibilityNativeComposition(CarriedRoleComposition):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.responsibility=ResponsibilityReadout(self.levels,self.width,self.head.out_features)
        with torch.no_grad():self.responsibility.head.load_state_dict(self.head.state_dict())
        self.last_responsibility=None

    def load_carried_state(self,weights):
        missing=self.load_state_dict(weights,strict=False)
        if missing.unexpected_keys or any(not k.startswith('responsibility.') for k in missing.missing_keys):
            raise ValueError('native carrier migration coverage changed')
        self.responsibility.head.load_state_dict(self.head.state_dict())
        return missing

    def compose(self,state,address=(0,0),pool_lower=True):
        _,time=super().compose(state,address,pool_lower)
        views=[];available=[];reads=0
        for level,(obj,bank) in enumerate(zip(self.objects,state.banks)):
            values=[]
            for child in sorted(bank.published):
                local=tuple(v-o for v,o in zip(child,state.frame_origin)) if level==0 else child
                ancestor=tuple(v//2**(self.levels-1-level) for v in local)
                if ancestor!=tuple(address):continue
                value,_=obj.read(bank,child,time);values.append(value);reads+=1
            available.append(bool(values))
            # Equal per-level access, not one output candidate per lower event.
            views.append(torch.stack(values).mean(0) if values else self.head.weight.new_zeros(self.width))
        logits,record=self.responsibility(torch.stack(views),torch.tensor(available,device=views[0].device))
        self.last_responsibility=record
        state.responsibility_reads=getattr(state,'responsibility_reads',0)+reads
        state.responsibility_projection_madds=getattr(state,'responsibility_projection_madds',0)+self.levels*self.width*self.width
        state.responsibility_head_madds=getattr(state,'responsibility_head_madds',0)+(self.levels+1)*self.width*self.head.out_features
        state.responsibility_query_madds=getattr(state,'responsibility_query_madds',0)+self.width*self.levels
        return logits,time

    def work(self,state):
        work=super().work(state)
        for key in ['responsibility_reads','responsibility_projection_madds','responsibility_head_madds','responsibility_query_madds']:
            work[key]=getattr(state,key,0)
        return work
