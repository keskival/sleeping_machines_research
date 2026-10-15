"""Persistent role innovations carried alongside native transformed memory.

Every parent still performs native receipt/update races. A bounded residual view
keeps its latest ordered child interaction available to indexed reads and native
emission races without requiring four more transformations to preserve it.
"""
import math
import torch
from torch import nn
from .relational_frame_composition import RelationalFrameComposition,RelationalEventCore
from .variance_reduced_native_composition import PureTimerObject


class CarriedRoleCore(RelationalEventCore):
    def new_state(self):
        state=super().new_state();state.role_carrier=None;state.role_carriers={}
        state.carrier_transport_calls=0;state.carrier_additions=0
        return state

    def consume_event(self,source,timestamp,content,state):
        captured=[]
        handle=self.composer.register_forward_hook(lambda module,args,out:captured.append(out))
        try:result=super().consume_event(source,timestamp,content,state)
        finally:handle.remove()
        if captured:
            # This is a composed published-child view, never a raw sensor skip.
            state.role_carrier=(captured[-1],torch.as_tensor(timestamp,dtype=torch.float64,device=captured[-1].device))
            state.role_carriers[state.events]=state.role_carrier
        return result


class CarriedRoleObject(PureTimerObject):
    def __init__(self,core,width,gain=.1):
        super().__init__(core,width)
        if abs(gain)>=1:raise ValueError('initial carrier gain must be inside (-1,1)')
        self.carrier_scale=nn.Parameter(torch.tensor(math.atanh(gain)))

    def publish(self,state,pending,now):
        super().publish(state,pending,now)
        private=state.private[pending.address]
        private.role_carriers={v:c for v,c in private.role_carriers.items() if v>=pending.version}

    def read(self,state,address,timestamp):
        value,version=super().read(state,address,timestamp)
        private=state.private[address]
        if version in private.role_carriers:
            role,time=private.role_carriers[version]
            age=torch.as_tensor(timestamp,dtype=torch.float64,device=role.device)-time
            if float(age.detach())<0:raise ValueError('carrier read precedes receipt')
            carried=torch.cat([self.core.transport(v,age,self.core.depth-1,h)
                for h,v in enumerate(role.split(self.core.payload))])
            value=value+torch.tanh(self.carrier_scale)*carried
            private.carrier_transport_calls+=self.core.heads;private.carrier_additions+=self.width
        return value,version


class CarriedRoleComposition(RelationalFrameComposition):
    def __init__(self,*args,carrier_gain=.5,**kwargs):
        super().__init__(*args,**kwargs)
        for level,obj in enumerate(self.objects[1:],1):
            core=CarriedRoleCore(obj.core.base,self.width,601+level)
            core.composer=obj.core.composer
            self.objects[level]=CarriedRoleObject(core,self.width,carrier_gain)

    def load_relational_state(self,weights):
        missing=self.load_state_dict(weights,strict=False)
        if missing.unexpected_keys or any(not k.endswith('.carrier_scale') for k in missing.missing_keys):
            raise ValueError('relational migration changed native/composer coverage')
        return missing

    def work(self,state):
        work=super().work(state)
        private=[p for bank in state.banks[1:] for p in bank.private.values()]
        work.update(carrier_transport_calls=sum(p.carrier_transport_calls for p in private),
                    carrier_additions=sum(p.carrier_additions for p in private),
                    carrier_tensor_bytes=sum(v.numel()*v.element_size()+t.numel()*t.element_size()
                         for p in private for v,t in p.role_carriers.values()))
        return work
