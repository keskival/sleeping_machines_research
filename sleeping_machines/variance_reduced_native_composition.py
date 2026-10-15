"""Native composition with winner-averaged pure emission-timer credit.

Parameters and stochastic forward computation are unchanged. Receipt races
still use their native coupled value/time teacher. Full event-order boundary
credit is separate; this operation reduces interior-gradient variance only.
"""
import math
import torch
from torch import nn
from .indexed_temporal_object import IndexedTemporalObject,MemoryTimer
from .native_memory_composition import NativeMemoryComposition
from .pure_race_timer import PureRaceTimer


class PureTimerObject(IndexedTemporalObject):
    def __init__(self,core,width):
        nn.Module.__init__(self)
        self.core,self.width=core,width

    def arm(self,state,address,now,head=0):
        value,version=self.read(state,address,now)
        query=self.core.queries[0][head](value)
        record=state.published[address]
        time=torch.as_tensor(now,dtype=torch.float64,device=value.device)
        visible=self.core.transport(record.values.split(self.core.payload)[head],
             time-record.times[head],self.core.depth-1,head)
        units=self.core.units[0][head][0]
        scores=torch.stack([(query @ (unit.key+unit.key_read(visible)))/math.sqrt(self.core.payload)
             +unit.clock_bias for unit in units]).clamp(-12,12)
        delay,winner=PureRaceTimer.apply(scores)
        state.timer_sequence+=1;state.timer_scores+=self.core.pool
        state.armed[address]=state.timer_sequence
        return MemoryTimer(address,version,time+delay,state.timer_sequence,int(winner))


class VarianceReducedNativeComposition(NativeMemoryComposition):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        # Reuse the actual native cores: no extra initialization, parameters or
        # RNG draws. State-dict keys and forward arrival schedules stay identical.
        self.objects=nn.ModuleList(PureTimerObject(obj.core,obj.width) for obj in self.objects)
