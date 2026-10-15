"""Indexed native temporal objects with lazy reads and race-generated emissions.

The consume_event transition below is adapted from AddressedEventHeads only to
retain incoming timestamp derivatives. Units, races, transport and shared rules
reuse SplitEventHeads. No benchmark-win inheritance is claimed for this adapter.
"""
from dataclasses import dataclass, field
import math
import torch
from torch import nn
from torch.nn import functional as F
from .split_event_heads import SplitEventHeads


class TensorTimedEventCore(SplitEventHeads):
    def consume_event(self, source, timestamp, content, state):
        source = int(source)
        input_time = torch.as_tensor(timestamp, dtype=torch.float64, device=self.embedding.weight.device)
        stamp = float(input_time.detach())
        if not 0 <= source < self.sources or not math.isfinite(stamp) or stamp < state.last_input_time:
            raise ValueError('Known observed address and finite nondecreasing timestamps required')
        mark = torch.as_tensor(content, dtype=self.embedding.weight.dtype,
                               device=self.embedding.weight.device)
        if mark.shape != (self.content_dim,) or not bool(torch.isfinite(mark).all()):
            raise ValueError('Finite fixed-width observed content required')
        x = self.embedding.weight[source]+self.content(mark)
        arrival = input_time
        if source in state.contexts:
            previous, times = state.contexts[source]
            # Admission is local to this source. Other sources keep independent
            # progress; no global periodic clock or global next-event barrier.
            ready = times.max()
            read_time = torch.maximum(arrival, ready)
            state.queue_wait_sum += float((read_time-arrival).detach())
            arrival = read_time
            context = self.align(list(previous.split(self.payload)), times, arrival, self.depth-1)
            x = F.layer_norm(x+torch.sigmoid(self.source_gate(x))*context, (self.total_payload,))
        for depth in range(self.depth):
            mixed = self.channel_mix[depth](x)
            all_features = F.layer_norm(mixed, (self.total_payload,))
            values, arrivals = [], []
            for head in range(self.heads):
                incoming = mixed[head*self.payload:(head+1)*self.payload]
                units = self.units[depth][head][source]
                query = self.queries[depth][head](all_features)
                scores = torch.stack([(query @ (unit.key+unit.key_read(
                    state.memories.get((depth,head,source,i), incoming.new_zeros(self.payload))))) /
                    math.sqrt(self.payload)+unit.clock_bias for i,unit in enumerate(units)]).clamp(-12,12)
                if self.training:
                    proposals = [unit.propose(incoming,state.memories.get((depth,head,source,i),
                        incoming.new_zeros(self.payload)),state.arrivals.get((depth,head,source,i)),arrival)
                        for i,unit in enumerate(units)]
                    value,delay,winner = self.race(scores,torch.stack([p[0] for p in proposals]))
                    index = int(winner); memory = proposals[index][1]
                    state.counterfactual_values += self.pool
                else:
                    _,delay,winner = self.race(scores)
                    index = int(winner)
                    value,memory = units[index].propose(incoming,state.memories.get((depth,head,source,index),
                        incoming.new_zeros(self.payload)),state.arrivals.get((depth,head,source,index)),arrival)
                address = (depth,head,source,index)
                state.memories[address] = memory; state.arrivals[address] = arrival
                state.visited_units.add(address)
                state.candidate_scores += self.pool; state.selected_updates += 1
                values.append(value); arrivals.append(arrival+delay)
            arrival = torch.stack(arrivals).max()
            x = self.align(values,arrivals,arrival,depth)
        state.contexts[source] = (torch.cat(values),torch.stack(arrivals))
        state.events += 1; state.last_input_time = stamp
        return self.head(x), arrival


@dataclass(frozen=True)
class PublishedMemory:
    version: int
    ready: torch.Tensor
    values: torch.Tensor
    times: torch.Tensor


@dataclass(frozen=True)
class PendingPublication:
    address: object
    version: int
    ready: torch.Tensor
    values: torch.Tensor
    times: torch.Tensor


@dataclass(frozen=True)
class MemoryTimer:
    address: object
    version: int
    due: torch.Tensor
    identity: int
    winner: int


@dataclass
class TemporalObjectState:
    private: dict = field(default_factory=dict)
    published: dict = field(default_factory=dict)
    issued: dict = field(default_factory=dict)
    seen: dict = field(default_factory=dict)
    armed: dict = field(default_factory=dict)
    timer_sequence: int = 0
    timer_scores: int = 0


class IndexedTemporalObject(nn.Module):
    """One shared native rule set, separately addressed persistent state.

accept computes a transition; publish makes it visible at completion. read has
no write/emission side effects. arm/fire add an independently scheduled native
race, without requiring input events or periodically updating dormant state.
"""
    def __init__(self, content_dim=16, payload=8, heads=2, depth=4, pool=2, protected_pairs=1):
        super().__init__()
        self.core=TensorTimedEventCore(sources=1,content_dim=content_dim,classes=payload*heads,
            payload=payload,heads=heads,depth=depth,pool=pool,credit='counterfactual',
            shared_maps=True,protected_pairs=protected_pairs)
        self.width=payload*heads

    def new_state(self):
        return TemporalObjectState()

    def accept(self,state,address,timestamp,content,event_identity):
        # Identity describes the source notification/query, not the receiver's
        # clock. Retries are idempotent; intentionally new computations have IDs.
        stream,sequence=event_identity
        if not isinstance(sequence,int) or sequence<0:
            raise ValueError('event identity requires stable stream key and nonnegative sequence')
        identity=(address,stream)
        if sequence<=state.seen.get(identity,-1):
            return None
        private=state.private.get(address)
        if private is None:
            private=self.core.new_state()
            state.private[address]=private
        _,ready=self.core.consume_event(0,timestamp,content,private)
        values,times=private.contexts[0]
        version=state.issued.get(address,0)+1
        state.issued[address]=version;state.seen[identity]=sequence
        return PendingPublication(address,version,ready,values,times)

    def publish(self,state,pending,now):
        if float(torch.as_tensor(now).detach()) < float(pending.ready.detach()):
            raise ValueError('cannot publish a future computation')
        previous=state.published.get(pending.address)
        if pending.version != (1 if previous is None else previous.version+1):
            raise ValueError('publication must preserve local version order')
        state.published[pending.address]=PublishedMemory(pending.version,pending.ready,
                                                        pending.values,pending.times)

    def read(self,state,address,timestamp):
        record=state.published[address]
        time=torch.as_tensor(timestamp,dtype=torch.float64,device=record.ready.device)
        if not bool(torch.isfinite(time)) or float(time.detach()) < float(record.ready.detach()):
            raise ValueError('read precedes published availability')
        values=list(record.values.split(self.core.payload))
        aligned=self.core.align(values,record.times,time,self.core.depth-1)
        return self.core.head(aligned),record.version

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
        _,delay,winner=self.core.race(scores)
        state.timer_sequence+=1;state.timer_scores+=self.core.pool
        time=torch.as_tensor(now,dtype=torch.float64,device=value.device)
        state.armed[address]=state.timer_sequence
        return MemoryTimer(address,version,time+delay,state.timer_sequence,int(winner))

    def fire(self,state,timer,now):
        if float(torch.as_tensor(now).detach()) < float(timer.due.detach()):
            raise ValueError('timer has not fired')
        if state.armed.get(timer.address)!=timer.identity:
            return None
        del state.armed[timer.address]
        if state.published[timer.address].version != timer.version:
            return None  # stale proposal; a new write may arm a new race
        return self.read(state,timer.address,timer.due)
