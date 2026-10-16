"""Native local layers with version-bound lower-memory queries.

Reuses the hierarchy BindingStore/PartBinding and ChildRoleComposer. Two query
slots are binding roles (not geometric quadrants or semantic concept labels).
All lower publication keys are scored; only selected and sampled alternative
values are decoded. Historical own-write traces accept local read credit;
producer input/route/version drift paths are not exact whole-network gradients.
"""
from collections import deque
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F
from .local_deep_race import LocalDeepRace
from .native_receiver_eligibility import ReceiverEligibility
from .sparse_training import _transport
from .bound_native_composition import BindingStore, BoundSnapshot, PartBinding
from .relational_frame_composition import ChildRoleComposer


@dataclass(frozen=True)
class LayerPublication:
    # Own receiver writes are numerical and immutable; no stored event graph.
    receivers: tuple
    inputs: tuple
    winners: tuple
    head_times: torch.Tensor
    key_view: torch.Tensor
    residual: torch.Tensor


class QueryableLocalDeepRace(LocalDeepRace):
    def __init__(self,*args,publication_capacity=16,**kwargs):
        super().__init__(*args,**kwargs)
        if publication_capacity < 2:raise ValueError('at least two lower versions required')
        self.publication_capacity=publication_capacity
        self.memory_queries=nn.ModuleList(nn.Linear(self.width,2*self.payload,bias=False) for _ in range(self.depth-1))
        self.memory_keys=nn.ModuleList(nn.Linear(self.width,self.payload,bias=False) for _ in range(self.depth-1))
        self.assemblers=nn.ModuleList(ChildRoleComposer(self.width,rank=min(16,self.width),seed=16160+d) for d in range(self.depth-1))

    def new_state(self):
        state=super().new_state()
        state.update(bindings=BindingStore(),publications=[deque() for _ in range(self.depth)],
            publication_serial=[0]*self.depth,memory_key_scores=0,memory_selected_reads=0,
            memory_alternative_reads=0,memory_transport_heads=0,memory_pair_products=0,
            queried_historical_versions=0,memory_gc_removed=0)
        return state

    def read_publication(self,state,key,now,packets,*,learn=True):
        entry=state['bindings'].entries[key];record=entry.native
        stamp=torch.as_tensor(now,dtype=torch.float64,device=record.key_view.device)
        if not bool(torch.isfinite(stamp)) or float(stamp.detach())<float(entry.available):
            raise ValueError('lower publication must be available before query')
        d=key[0];values=[]
        for h,(u,trace,incoming) in enumerate(zip(record.winners,record.receivers,record.inputs)):
            unit=self.units[d][h][u];memory=trace.memory().requires_grad_(learn)
            if learn:packets.append((unit,trace,memory))
            value=self.value(unit,incoming,memory)
            values.append(_transport(value[None],(stamp-record.head_times[h]).reshape(1),
                self.transport_rate[d,h],self.transport_frequency[d,h])[0])
        state['bindings'].reads+=1;state['memory_transport_heads']+=self.heads
        return torch.nn.functional.layer_norm(record.residual+torch.cat(values)/math.sqrt(self.depth),(self.width,))

    def query_lower(self,d,x,arrival,state,generator,alternative_generator,packets,*,learn):
        candidates=list(state['publications'][d-1])
        if not candidates:return x,arrival,()
        views=torch.stack([state['bindings'].entries[k].native.key_view for k in candidates])
        scores=(self.memory_queries[d-1](x).reshape(2,self.payload) @
            self.memory_keys[d-1](views).T/math.sqrt(self.payload)).clamp(-12,12)
        state['memory_key_scores']+=scores.numel();remaining=list(range(len(candidates)))
        chosen=[];times=[];parts=[]
        for role in range(min(2,len(candidates))):
            available=torch.tensor(remaining,dtype=torch.long,device=x.device)
            logits=scores[role,available];rates=logits.double().exp()
            noise=torch.empty(len(remaining),dtype=torch.float64,device=x.device).exponential_(generator=generator)
            first,local=(noise/rates.detach()).min(0);j=remaining[int(local)];key=candidates[j]
            value=self.read_publication(state,key,arrival,packets,learn=learn)
            state['memory_selected_reads']+=1
            state['queried_historical_versions']+=int(key!=candidates[-1])
            if learn and len(remaining)>1:
                pi=rates/rates.sum();alt_noise=torch.empty_like(rates).exponential_(generator=alternative_generator)
                alt_times=(alt_noise/pi.detach());alt_times[int(local)]=float('inf');a=int(alt_times.argmin())
                with torch.no_grad():other=self.read_publication(state,candidates[remaining[a]],arrival,[],learn=False)
                coefficient=(pi[a]-pi[a].detach())/(pi[a]/(1-pi[int(local)])).detach()
                value=value+coefficient.to(value.dtype)*(other-value).detach()
                state['memory_alternative_reads']+=1
            clock=first-first*(rates.sum().log()-rates.sum().log().detach())
            times.append(arrival+.001+.010*clock/(1+clock))
            chosen.append(value);parts.append(PartBinding(key,arrival.detach().clone(),(float(role),0.),'query'))
            remaining.remove(j)
        ready=torch.stack(times).max()
        # Each selected query response has its own causal race completion time.
        aligned={role:torch.cat([_transport(v[None],(ready-times[role]).reshape(1),
            self.transport_rate[d,h],self.transport_frequency[d,h])[0]
            for h,v in enumerate(value.split(self.payload))]) for role,value in enumerate(chosen)}
        state['memory_transport_heads']+=len(chosen)*self.heads
        if len(chosen)==2:state['memory_pair_products']+=self.assemblers[d-1].rank
        return x+self.assemblers[d-1](aligned),ready,tuple(parts)

    def publish_layer(self,d,state,arrival,head_times,winners,parts,mixed,stream,x):
        state['publication_serial'][d]+=1
        key=(d,tuple(winners),state['publication_serial'][d])
        traces=tuple(state['receivers'][(d,h,u)] for h,u in enumerate(winners))
        incoming=tuple(v.detach().clone() for v in mixed.split(self.payload))
        record=LayerPublication(traces,incoming,tuple(winners),torch.stack(head_times).detach().clone(),x.detach().clone(),stream.detach().clone())
        state['bindings'].put(BoundSnapshot(key,arrival.detach().clone(),record,None,parts))
        bank=state['publications'][d];bank.append(key)
        while len(bank)>self.publication_capacity:bank.popleft()

    def storage(self,state):
        result=super().storage(state)
        records=list(state['bindings'].entries.values())
        traces={id(t):t for e in records for t in e.native.receivers}
        result.update(bound_publications=len(records),bound_edges=sum(len(e.parts) for e in records),
            publication_tensor_bytes=sum(e.available.numel()*e.available.element_size()+
                e.native.key_view.numel()*e.native.key_view.element_size()+e.native.residual.numel()*e.native.residual.element_size()+e.native.head_times.numel()*e.native.head_times.element_size()+
                sum(v.numel()*v.element_size() for v in e.native.inputs) for e in records),
            retained_publication_trace_bytes=sum(t.storage()['tensor_bytes'] for t in traces.values()),
            scope='Numerical native receiver eligibility plus immutable publication inputs/keys/traces and transitive bound parts. Trace bytes may alias current receiver traces;Python metadata,current graphs and credit packets additional. Per-layer root rings bounded;transitive lineage is not constant memory.')
        return result

    def consume(self,message,state,generator,alternative_generator,*,learn=True,noise=None,alternative_noise=None,cause_messages=None):
        if message.shape!=(self.width,) or not bool(torch.isfinite(message).all()):
            raise ValueError('finite fixed-width native event message required')
        if message.device.type!='cpu':raise ValueError('initial event kernel is CPU-only')
        if cause_messages is not None and cause_messages.shape!=(self.depth,self.width):
            raise ValueError('one cause-message inbox per native depth required')
        x=message;arrival=torch.tensor(float(state['events']),dtype=torch.float64)
        outputs=[];packets=[];winners=[]
        for d in range(self.depth):
            # Explicit local interface: numerical information persists, gradient
            # graphs do not cross a message boundary into a different layer.
            if d:x=x.detach();arrival=arrival.detach()
            parts=()
            if d:x,arrival,parts=self.query_lower(d,x,arrival,state,generator,alternative_generator,packets,learn=learn)
            if cause_messages is not None:x=x+cause_messages[d].detach()
            stream=x;mixed=self.mix[d](x);features=F.layer_norm(mixed,(self.width,))
            values=[];times=[];level_winners=[]
            for h in range(self.heads):
                units=self.units[d][h];incoming=mixed[h*self.payload:(h+1)*self.payload]
                q=self.query[d][h](features);old=[];memories=[];scores=[]
                for u,unit in enumerate(units):
                    trace=state['receivers'].get((d,h,u))
                    if trace is None:trace=ReceiverEligibility.empty(unit)
                    old.append(trace);m=trace.memory().requires_grad_(learn);memories.append(m)
                    scores.append(q@(unit.key+unit.key_read(m))/math.sqrt(self.payload)+unit.clock_bias)
                    if learn:packets.append((unit,trace,m))
                scores=torch.stack(scores).clamp(-12,12);rates=scores.double().exp()
                draws=(torch.empty(self.pool,dtype=torch.float64).exponential_(generator=generator)
                       if noise is None else noise[d,h])
                first,winner=(draws/rates.detach()).min(0);u=int(winner);unit=units[u]
                proposed=old[u].propose(unit,incoming,arrival.detach())
                memory=proposed.memory().requires_grad_(learn)
                value=self.value(unit,incoming,memory)
                if learn:
                    packets.append((unit,proposed,memory))
                    if self.pool>1:
                        pi=rates/rates.sum();alt_noise=(torch.empty(self.pool,dtype=torch.float64).exponential_(generator=alternative_generator)
                            if alternative_noise is None else alternative_noise[d,h])
                        alt_times=alt_noise/pi.detach();alt_times[u]=float('inf');j=int(alt_times.argmin())
                        with torch.no_grad():
                            stamp=None if old[j].timestamp is None else arrival.new_tensor(old[j].timestamp)
                            alternative,_=units[j].propose(incoming.detach(),old[j].memory(),stamp,arrival.detach())
                        probability=pi[j]/(1-pi[u]);coefficient=((pi[j]-pi[j].detach())/probability.detach()).to(value.dtype)
                        value=value+coefficient*(alternative-value).detach()
                        state['alternative_values']+=1
                # Native factorized-clock local credit; no graph of past clocks.
                clock=first-first*(rates.sum().log()-rates.sum().log().detach())
                delay=.001+.010*clock/(1+clock)
                values.append(value);times.append(arrival+delay)
                state['receivers'][(d,h,u)]=proposed
                state['candidate_scores']+=self.pool;state['selected_updates']+=1;level_winners.append(u)
            ready=torch.stack(times).max()
            aligned=torch.cat([_transport(v[None],(ready-t).reshape(1),self.transport_rate[d,h],self.transport_frequency[d,h])[0] for h,(v,t) in enumerate(zip(values,times))])
            x=F.layer_norm(stream+aligned/math.sqrt(self.depth),(self.width,));arrival=ready
            self.publish_layer(d,state,arrival,times,level_winners,parts,mixed,stream,x)
            outputs.append(x);winners.append(level_winners)
        state['events']+=1
        roots=[key for bank in state['publications'] for key in bank]
        state['memory_gc_removed']+=state['bindings'].collect(roots)
        return torch.stack(outputs),state,packets,dict(winners=winners,ready=float(arrival.detach()))
