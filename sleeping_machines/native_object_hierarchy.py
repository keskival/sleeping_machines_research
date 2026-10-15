"""Thin causal dispatcher over indexed native temporal objects.

One object primitive provides state, indexed reads and generated events. The
heap is a chronological reference executor, not a measured async backend.
"""
from dataclasses import dataclass,field
import heapq,math
import torch
from torch import nn
from .indexed_temporal_object import IndexedTemporalObject


@dataclass
class ObjectHierarchyState:
    banks: list
    queue: list = field(default_factory=list)
    inputs: dict = field(default_factory=dict)
    input_issued: dict = field(default_factory=dict)
    serial: int = 0
    now: float = 0.
    deliveries: int = 0
    publications: int = 0
    queries: int = 0
    timer_emissions: int = 0
    indexed_reads: int = 0


class NativeObjectHierarchy(nn.Module):
    """Eight abstraction levels, each with four native temporal race stages.

Spatial ownership is initially geometric. Higher levels may query any indexed
lower memory; query selection and autonomous timer-arming policies are not yet
learned. Native state/races/transport are reused, not replaced by pooling blocks.
"""
    def __init__(self,input_width=2,payload=8,heads=4,levels=8,depth=4,pool=2,classes=11):
        super().__init__()
        self.width,self.levels=payload*heads,levels
        self.input=nn.Linear(input_width,self.width)
        self.objects=nn.ModuleList(IndexedTemporalObject(self.width+6,payload,heads,depth,pool,
                 protected_pairs=1 if payload>=4 else 0) for _ in range(levels))
        self.head=nn.Linear(self.width,classes)

    def new_state(self):
        return ObjectHierarchyState([obj.new_state() for obj in self.objects])

    def _enqueue(self,state,time,kind,data):
        stamp=float(torch.as_tensor(time).detach())
        if not math.isfinite(stamp) or stamp<state.now:
            raise ValueError('no scheduling into the observed past')
        state.serial+=1;heapq.heappush(state.queue,(stamp,state.serial,time,kind,data))

    def observe(self,state,address,timestamp,content):
        address=tuple(address)
        value=self.input(torch.as_tensor(content,dtype=self.input.weight.dtype,device=self.input.weight.device))
        self._enqueue(state,timestamp,'input',(address,value))

    def _send(self,state,source_level,source_address,value,version,time,kind,target=None):
        level=source_level+1 if target is None else target[0]
        if level>=self.levels:return
        address=tuple(v//2 for v in source_address) if target is None else tuple(target[1])
        source_scale=2**(source_level+1);target_scale=2**(level+1)
        pose=[(source_address[i]*source_scale-address[i]*target_scale)/target_scale for i in range(2)]
        types={'write':[1.,0.,0.],'query':[0.,1.,0.],'timer':[0.,0.,1.]}
        metadata=value.new_tensor([*pose,source_level/max(1,self.levels),*types[kind]])
        content=torch.cat((value,metadata))
        identity=((source_level,tuple(source_address),kind),version)
        self._enqueue(state,time,'deliver',(level,address,content,identity))

    def advance(self,state,until):
        until=float(until)
        if not math.isfinite(until) or until<state.now:raise ValueError('causal cutoff required')
        while state.queue and state.queue[0][0]<=until:
            stamp,_,time,kind,data=heapq.heappop(state.queue);state.now=stamp
            if kind=='input':
                address,value=data
                version=state.input_issued.get(address,0)+1;state.input_issued[address]=version
                state.inputs[address]=(value,version,stamp)
                self._send(state,-1,address,value,version,time,'write')
            elif kind=='deliver':
                level,address,content,identity=data
                pending=self.objects[level].accept(state.banks[level],address,time,content,identity)
                if pending is not None:
                    state.deliveries+=1;self._enqueue(state,pending.ready,'publish',(level,pending))
            elif kind=='publish':
                level,pending=data;obj=self.objects[level];bank=state.banks[level]
                obj.publish(bank,pending,time);state.publications+=1
                value,version=obj.read(bank,pending.address,time)
                self._send(state,level,pending.address,value,version,time,'write')
            elif kind=='timer':
                level,timer=data
                fired=self.objects[level].fire(state.banks[level],timer,time)
                if fired is not None:
                    value,_=fired;state.timer_emissions+=1
                    self._send(state,level,timer.address,value,timer.identity,time,'timer')
        state.now=until
        return state

    def read(self,state,level,address):
        address=tuple(address);state.indexed_reads+=1
        if level==-1:return state.inputs[address][:2]
        return self.objects[level].read(state.banks[level],address,state.now)

    def arm(self,state,level,address):
        timer=self.objects[level].arm(state.banks[level],tuple(address),state.now)
        self._enqueue(state,timer.due,'timer',(level,timer))
        return timer

    def query(self,state,source_level,source_address,target_level,target_address):
        if not source_level<target_level:raise ValueError('this query interface reads lower memories')
        value,_=self.read(state,source_level,source_address);state.queries+=1
        # Query sequence is computational identity, not a new external observation.
        self._send(state,source_level,tuple(source_address),value,state.queries,state.now,'query',
                   target=(target_level,tuple(target_address)))

    def logits(self,state,address=(0,0)):
        value,_=self.read(state,self.levels-1,address)
        return self.head(value)

    def work(self,state):
        private=[p for bank in state.banks for p in bank.private.values()]
        return dict(addressed_objects=len(private),external_observations=sum(state.input_issued.values()),
            candidate_scores=sum(p.candidate_scores for p in private),
            selected_updates=sum(p.selected_updates for p in private),
            counterfactual_value_evaluations=sum(p.counterfactual_values for p in private),
            timer_candidate_scores=sum(b.timer_scores for b in state.banks),
            deliveries=state.deliveries,publications=state.publications,
            indexed_reads=state.indexed_reads,queries=state.queries,timer_emissions=state.timer_emissions,
            state_tensor_bytes=sum(p.storage()['persistent_tensor_bytes'] for p in private),
            boundary='Native state tensors only; published aliases, input records, queue, Python metadata, parameters and autograd graphs separate. Operation counts, not complete FLOPs/energy.')
