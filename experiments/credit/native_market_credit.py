"""Event-native market credit with persistent addressed state and causal history."""
import copy
from dataclasses import dataclass
import numpy as np
import torch
from sleeping_machines.addressed_event_heads import AddressedEventHeads
from sleeping_machines.route_activation_memory import RouteActivationMemory

@dataclass
class NativeCarry:
    state: object
    history: object
    time: float
    version: int
    def detach(self):
        state=copy.copy(self.state)
        state.memories={k:v.detach() for k,v in state.memories.items()}
        state.arrivals={k:v.detach() for k,v in state.arrivals.items()}
        state.contexts={k:(v.detach(),t.detach()) for k,(v,t) in state.contexts.items()}
        state.visited_units=set(state.visited_units)
        return NativeCarry(state,copy.deepcopy(self.history),self.time,self.version)

class NativeMarketCredit(torch.nn.Module):
    def __init__(self,width):
        super().__init__();self.width=width;self.norm=torch.nn.LayerNorm(width)
        self.core=AddressedEventHeads(sources=7,content_dim=width+8,classes=2*(width//3),payload=8,depth=2,pool=2,heads=2)
        self.history_adapter=torch.nn.Linear(2*width+14,self.core.total_payload,bias=False)
        torch.nn.init.zeros_(self.history_adapter.weight)
        torch.nn.init.zeros_(self.core.head.weight);torch.nn.init.zeros_(self.core.head.bias)
        self.events=0;self.selected_updates=0;self.candidate_scores=0;self.counterfactual_values=0
    def forward(self,start,events,carried):
        assert start.shape==(1,self.width) and events.shape[0]==1
        if isinstance(carried,NativeCarry):c=carried.detach()
        else:c=NativeCarry(self.core.new_state(),RouteActivationMemory(self.width,np.eye(4,self.width)),0.,0)
        # One shared input projection, native nonlinear temporal heads throughout.
        token=torch.cat((self.norm(start[0]),start.new_zeros(8)))
        _,_=self.core.consume_event(6,c.time,token,c.state)
        for event in events[0]:
            feedback=event[-8:];c.time+=float(feedback[0].expm1().clamp_min(0).detach())
            source=int(feedback[1:7].argmax());features=event[:-8]
            snapshot=c.history.snapshot(source,c.time,str(c.version))
            hist=features.new_zeros(2*self.width+14)
            if snapshot is not None:
                lower=np.tril_indices(4)
                extra=np.concatenate((snapshot.mean,snapshot.variance,snapshot.projected_covariance[lower],np.array([np.log1p(snapshot.observations),np.log1p(snapshot.effective_count),np.log1p(snapshot.age),float(snapshot.operator_stale)])))
                hist=torch.as_tensor(extra,dtype=features.dtype,device=features.device)
            # History adds a detached observation token through a learned adapter.
            extra=self.history_adapter(hist)
            # Add typed history to content through the core's public projection
            # without mutating parameters or substituting its native state.
            token=torch.cat((self.norm(features),feedback))
            # The core expects raw content; a hook inserts the same additive token.
            hook=self.core.content.register_forward_hook(lambda module,args,out:out+extra)
            try:output,_=self.core.consume_event(source,c.time,token,c.state)
            finally:hook.remove()
            c.history.observe(source,features.detach().cpu().numpy(),c.time,str(c.version))
        c.version+=1
        self.events+=len(events[0])+1
        self.selected_updates=c.state.selected_updates;self.candidate_scores=c.state.candidate_scores;self.counterfactual_values=c.state.counterfactual_values
        return output[None],c
