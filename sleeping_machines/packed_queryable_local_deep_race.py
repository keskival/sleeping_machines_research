"""Packed execution of the version-bound native query/composition reference."""
import copy
from collections import deque
from dataclasses import dataclass
import math
import torch
from torch.nn import functional as F
from .packed_local_deep_race import PackedLocalDeepRace,FIELDS,TRACES
from .native_receiver_eligibility import ReceiverEligibility
from .sparse_training import _transport
from .bound_native_composition import BindingStore,BoundSnapshot
from .queryable_local_deep_race import QueryableLocalDeepRace,LayerPublication


@dataclass
class HistoricalReadCredit:
    depth: int
    traces: tuple
    memory: torch.Tensor
    winners: tuple


class PackedQueryableLocalDeepRace(PackedLocalDeepRace):
    query_lower=QueryableLocalDeepRace.query_lower

    def __init__(self,source):
        super().__init__(source)
        self.publication_capacity=source.publication_capacity
        self.memory_queries=copy.deepcopy(source.memory_queries)
        self.memory_keys=copy.deepcopy(source.memory_keys)
        self.assemblers=copy.deepcopy(source.assemblers)

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
        d=key[0];hi=torch.arange(self.heads,device=record.key_view.device)
        winners=torch.tensor(record.winners,dtype=torch.long,device=hi.device)
        params={name:bank[d,hi,winners] for name,bank in self.banks.items()}
        memory=torch.stack([t.memory() for t in record.receivers]).requires_grad_(learn)
        if learn:packets.append(HistoricalReadCredit(d,record.receivers,memory,record.winners))
        values=self.unit_value(params,torch.stack(record.inputs),memory)
        age=stamp-record.head_times
        magnitude=torch.exp(-age.to(values.dtype)[:,None]*(F.softplus(self.transport_rate[d])+1e-6))
        angle=age[:,None]*self.transport_frequency[d].double();pairs=values.reshape(self.heads,-1,2)
        real,imag=pairs[...,0]*magnitude,pairs[...,1]*magnitude
        c,s=angle.cos().to(values.dtype),angle.sin().to(values.dtype)
        aligned=torch.stack((c*real-s*imag,s*real+c*imag),-1).flatten()
        state['bindings'].reads+=1;state['memory_transport_heads']+=self.heads
        return F.layer_norm(record.residual+aligned/math.sqrt(self.depth),(self.width,))

    def publish_layer(self,d,state,arrival,head_times,winners,parts,mixed,stream,x):
        state['publication_serial'][d]+=1;winners=tuple(int(u) for u in winners)
        key=(d,winners,state['publication_serial'][d]);old=state['layers'][d]
        # Clone selected slices: retaining views would pin full obsolete banks.
        traces=tuple(ReceiverEligibility(*(old[name][h,u].detach().clone()
            for name in ['value',*TRACES]),float(old['time'][h,u])) for h,u in enumerate(winners))
        record=LayerPublication(traces,tuple(v.detach().clone() for v in mixed.split(self.payload)),
            winners,head_times.detach().clone(),x.detach().clone(),stream.detach().clone())
        state['bindings'].put(BoundSnapshot(key,arrival.detach().clone(),record,None,parts))
        bank=state['publications'][d];bank.append(key)
        while len(bank)>self.publication_capacity:bank.popleft()

    @torch.no_grad()
    def deliver_credit(self,packets,state):
        super().deliver_credit([p for p in packets if not isinstance(p,HistoricalReadCredit)],state)
        mapping={path:name for name,path in FIELDS.items()}
        for packet in packets:
            if not isinstance(packet,HistoricalReadCredit) or packet.memory.grad is None:continue
            for h,(u,trace,g) in enumerate(zip(packet.winners,packet.traces,packet.memory.grad)):
                for path,credit in trace.credit(g).items():
                    bank=self.banks[mapping[path]]
                    if bank.grad is None:bank.grad=torch.zeros_like(bank)
                    bank.grad[packet.depth,h,u]+=credit
                state['credit_projections']+=1

    def consume(self,message,state,generator,alternative_generator,*,learn=True,noise=None,alternative_noise=None,cause_messages=None):
        h,u,p=self.heads,self.pool,self.payload;hi=torch.arange(h);x=message;arrival=torch.tensor(float(state['events']),dtype=torch.float64)
        outputs=[];packets=[];winners=[]
        if cause_messages is not None and cause_messages.shape!=(self.depth,self.width):
            raise ValueError('one cause-message inbox per native depth required')
        for d in range(self.depth):
            if d:x=x.detach();arrival=arrival.detach()
            parts=()
            if d:x,arrival,parts=self.query_lower(d,x,arrival,state,generator,alternative_generator,packets,learn=learn)
            if cause_messages is not None:x=x+cause_messages[d].detach()
            stream=x;mixed=F.linear(x,self.mix[d]);normal=F.layer_norm(mixed,(self.width,));incoming=mixed.reshape(h,p)
            input_arrival=arrival.detach()
            old=state['layers'][d];b={k:v[d] for k,v in self.banks.items()}
            memory=torch.view_as_real(old['value']).flatten(-2).detach().requires_grad_(learn)
            keys=b['key']+torch.einsum('hupq,huq->hup',b['key_read'],memory)
            query=torch.einsum('hpw,w->hp',self.query[d],normal)
            scores=((query[:,None]*keys).sum(-1)/math.sqrt(p)+b['clock_bias']).clamp(-12,12);rates=scores.double().exp()
            draws=torch.empty(h,u,dtype=torch.float64).exponential_(generator=generator) if noise is None else noise[d]
            first,winner=(draws/rates.detach()).min(-1);selected={k:v[hi,winner] for k,v in b.items()}
            previous={k:v[hi,winner] for k,v in old.items()}
            elapsed=torch.where(previous['seen'],arrival.detach()-previous['time'],torch.zeros_like(previous['time']))
            proposed=self.proposal(previous,selected,incoming,elapsed,traces=True)
            new_memory=torch.view_as_real(proposed['value']).flatten(-2).detach().requires_grad_(learn)
            value=self.unit_value(selected,incoming,new_memory)
            if learn:
                packets.append((d,old,memory,proposed,new_memory,winner))
                if u>1:
                    pi=rates/rates.sum(-1,keepdim=True)
                    draws_alt=torch.empty(h,u,dtype=torch.float64).exponential_(generator=alternative_generator) if alternative_noise is None else alternative_noise[d]
                    alt_times=(draws_alt/pi.detach()).scatter(1,winner[:,None],float('inf'));alt=alt_times.argmin(-1)
                    with torch.no_grad():
                        ap={k:v[hi,alt] for k,v in b.items()};ast={k:v[hi,alt] for k,v in old.items()}
                        adt=torch.where(ast['seen'],arrival.detach()-ast['time'],torch.zeros_like(ast['time']))
                        am=self.proposal(ast,ap,incoming,adt,traces=False)['value']
                        alternative=self.unit_value(ap,incoming,torch.view_as_real(am).flatten(-2))
                    pa=pi[hi,alt];pw=pi[hi,winner];coefficient=((pa-pa.detach())/(pa/(1-pw)).detach()).to(value.dtype)
                    value=value+coefficient[:,None]*(alternative-value).detach();state['alternative_values']+=h
            log_rate=rates.sum(-1).log();clock=first-first*(log_rate-log_rate.detach())
            times=arrival+.001+.010*clock/(1+clock);ready=times.max();age=ready-times
            rate=F.softplus(self.transport_rate[d])+1e-6;mag=torch.exp(-age.to(x.dtype)[:,None]*rate)
            angle=age[:,None]*self.transport_frequency[d].double();pairs=value.reshape(h,-1,2)
            real,imag=pairs[...,0]*mag,pairs[...,1]*mag;c,s=angle.cos().to(x.dtype),angle.sin().to(x.dtype)
            aligned=torch.stack((c*real-s*imag,s*real+c*imag),-1).flatten()
            x=F.layer_norm(stream+aligned/math.sqrt(self.depth),(self.width,));arrival=ready
            mask=F.one_hot(winner,u).bool();updated={}
            for name in ['value',*TRACES]:
                v=old[name];shape=mask.shape+(1,)*(v.ndim-2)
                updated[name]=torch.where(mask.reshape(shape),proposed[name][:,None],v)
            # Receiver writes retain stage input time, not emission time.
            updated['time']=torch.where(mask,input_arrival.expand(h,u),old['time']);updated['seen']=old['seen']|mask
            state['layers'][d]=updated;state['candidate_scores']+=h*u;state['selected_updates']+=h
            self.publish_layer(d,state,arrival,times,winner,parts,mixed,stream,x)
            outputs.append(x);winners.append(winner.detach().tolist())
        state['events']+=1
        state['memory_gc_removed']+=state['bindings'].collect([key for bank in state['publications'] for key in bank])
        return torch.stack(outputs),state,packets,dict(winners=winners,ready=float(arrival.detach()))

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
