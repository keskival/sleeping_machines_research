"""Bounded delayed content credit through actual version-bound query children.

Replays a finite local producer DAG, never the whole event history. Factual
values/RNGs are unchanged. Conditional on recorded choices, timings and writes;
changing-parameter replay remains approximate. Historical route decisions and
base-input/inbox provenance are outside this added credit path.
"""
from dataclasses import dataclass
import math
import torch
from torch.nn import functional as F
from .coupled_native_query_credit import CoupledNativeQueryCredit
from .sparse_training import _transport


@dataclass(frozen=True)
class CompositionalLineage:
    sources: tuple
    issued: tuple
    reply_times: tuple
    ready: torch.Tensor


class CompositionalProducerCredit(CoupledNativeQueryCredit):
    replay_hops=2
    replay_edge_budget=6

    def new_state(self):
        state=super().new_state()
        state.update(compositional_lineage={},recursive_replay_roots=0,
            recursive_replay_edges=0,recursive_replay_truncated=0)
        return state

    def align_replies(self,d,values,times):
        ready,views=super().align_replies(d,values,times)
        if getattr(self,'_capturing_lineage',False) and self._first_alignment is None:
            self._first_alignment=(tuple(t.detach().clone() for t in times),ready.detach().clone())
        return ready,views

    def query_lower(self,d,x,arrival,state,generator,alternative_generator,packets,*,learn):
        self._capturing_lineage=True;self._first_alignment=None
        try:
            result=super().query_lower(d,x,arrival,state,generator,alternative_generator,packets,learn=learn)
            if self._first_alignment is not None:
                times,ready=self._first_alignment;parts=result[2]
                self._lineage_inputs[d]=CompositionalLineage(tuple(p.source for p in parts),
                    tuple(p.issued.detach().clone() for p in parts),times,ready)
            return result
        finally:
            self._capturing_lineage=False;self._first_alignment=None

    def publish_layer(self,d,state,*args):
        super().publish_layer(d,state,*args)
        key=state['publications'][d][-1]
        if d in self._lineage_inputs:state['compositional_lineage'][key]=self._lineage_inputs[d]

    def consume(self,*args,**kwargs):
        self._lineage_inputs={}
        try:
            result=super().consume(*args,**kwargs)
            state=result[1];live=state['bindings'].entries
            state['compositional_lineage']={k:v for k,v in state['compositional_lineage'].items() if k in live}
            return result
        finally:del self._lineage_inputs

    def read_publication(self,state,key,now,packets,*,learn=True):
        root=not hasattr(self,'_composition_credit_context')
        if root:
            self._composition_credit_context=dict(hops=self.replay_hops,remaining=self.replay_edge_budget,packets=packets,learn=learn)
            if learn:state['recursive_replay_roots']+=1
        try:return super().read_publication(state,key,now,packets,learn=learn)
        finally:
            if root:del self._composition_credit_context

    def aligned_child(self,d,value,lineage,role):
        age=(lineage.ready-lineage.reply_times[role]).reshape(1)
        return torch.cat([_transport(v[None],age,self.transport_rate[d,h],
            self.transport_frequency[d,h])[0] for h,v in enumerate(value.split(self.payload))])

    def producer_inputs(self,state,key):
        ctx=self._composition_credit_context;lineage=state['compositional_lineage'].get(key)
        producer=state['producer_records'].get(key)
        if not ctx['learn'] or lineage is None or producer is None:
            return super().producer_inputs(state,key)
        if ctx['hops']<=0 or ctx['remaining']<=0:
            state['recursive_replay_truncated']+=len(lineage.sources)
            return super().producer_inputs(state,key)
        views=dict(producer.views);d=key[0]
        for role,child in enumerate(lineage.sources):
            if ctx['remaining']<=0:
                state['recursive_replay_truncated']+=1;continue
            if child[0]>=d:raise ValueError('producer lineage must descend in abstraction level')
            ctx['remaining']-=1;ctx['hops']-=1
            try:value=self.read_publication(state,child,lineage.issued[role],ctx['packets'],learn=True)
            finally:ctx['hops']+=1
            aligned=self.aligned_child(d,value,lineage,role)
            # Original content stays bitwise; only its bounded conditional VJP is added.
            views[role]=views[role]+(aligned-aligned.detach())
            state['recursive_replay_edges']+=1;state['memory_transport_heads']+=self.heads
        record=state['bindings'].entries[key].native
        stream=producer.base_input+producer.inbox+self.assemblers[d-1](views)
        mixed=F.linear(stream,self.mix[d]).reshape(self.heads,self.payload)
        return record.residual+(stream-stream.detach()),torch.stack(record.inputs)+(mixed-mixed.detach()),producer

    def storage(self,state):
        result=super().storage(state)
        result.update(compositional_lineage_records=len(state['compositional_lineage']),
            compositional_lineage_tensor_bytes=sum(sum(t.numel()*t.element_size() for t in [*r.issued,*r.reply_times,r.ready]) for r in state['compositional_lineage'].values()),
            recursive_replay_hops=self.replay_hops,recursive_replay_edge_budget=self.replay_edge_budget)
        return result


def enable_compositional_credit(model,state,*,hops=2,edge_budget=6):
    if type(model.core) is not CoupledNativeQueryCredit:raise ValueError('legal joint-query parent required')
    if hops<0 or edge_budget<0:raise ValueError('nonnegative replay limits required')
    ids=[id(p) for p in model.parameters()];model.core.__class__=CompositionalProducerCredit
    model.core.replay_hops=hops;model.core.replay_edge_budget=edge_budget
    for name,value in [('compositional_lineage',{}),('recursive_replay_roots',0),('recursive_replay_edges',0),('recursive_replay_truncated',0)]:state.setdefault(name,value)
    assert ids==[id(p) for p in model.parameters()]
    return dict(new_parameters=0,parameter_objects_preserved=True,replay_hops=hops,replay_edge_budget=edge_budget,
        scope='Bounded actual-child content VJP; old publications retained without invented lineage. Historical route/clock choices, base-input and inbox provenance remain fixed; changed-parameter replay approximate.')
