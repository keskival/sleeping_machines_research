"""Sampled version-bound VJP through direct messages AND queried children.

Numerical forward values, factual races and parameter objects are preserved.
This closes the missing depth edge, not unrecorded token/inbox history or
historical categorical choices. Replay at changing parameters is conditional.
"""
from dataclasses import dataclass
import torch
from torch.nn import functional as F
from .sampled_compositional_credit import SampledCompositionalCredit
from .bound_native_composition import PartBinding
from .producer_replay_native_race import local_write

@dataclass(frozen=True)
class DirectProducerEdge:
    source: tuple
    issued: torch.Tensor

class DirectInputProducerCredit(SampledCompositionalCredit):
    def new_state(self):
        state=super().new_state();state.update(direct_producer_edges={},direct_producer_replays=0,immediate_direct_replays=0,current_write_input_vjps=0)
        return state

    def publish_layer(self,d,state,arrival,head_times,winners,parts,mixed,stream,x):
        edge=None
        if d:
            child=state['publications'][d-1][-1]
            edge=DirectProducerEdge(child,state['bindings'].entries[child].available.detach().clone())
            # A separate use may alias a queried child; GC must retain both
            # uses' source even when the query ring later evicts that version.
            parts=(*parts,PartBinding(edge.source,edge.issued,(0.,0.),'direct_message'))
        super().publish_layer(d,state,arrival,head_times,winners,parts,mixed,stream,x)
        if edge is not None:state['direct_producer_edges'][state['publications'][d][-1]]=edge

    def proposal(self,old,params,incoming,elapsed,*,traces):
        result=super().proposal(old,params,incoming,elapsed,traces=traces)
        if traces:
            names=['control_w','control_b','input','raw_rate','frequency']
            self._current_input_write=(incoming,old['value'].detach(),{n:params[n].detach() for n in names},elapsed.detach())
        return result

    def unit_value(self,params,incoming,memory):
        context=getattr(self,'_current_input_write',None)
        if torch.is_grad_enabled() and context is not None and incoming is context[0]:
            replay=local_write(context[1],context[2],incoming,context[3])
            memory=memory+(replay-replay.detach())
            self._current_write_vjps=getattr(self,'_current_write_vjps',0)+1
        return super().unit_value(params,incoming,memory)

    def attach_direct_message(self,d,x,arrival,state,packets,*,learn):
        if learn and d:
            key=state['publications'][d-1][-1]
            value=self.read_publication(state,key,state['bindings'].entries[key].available,packets,learn=True)
            x=x+(value-value.detach())
            state['immediate_direct_replays']=state.get('immediate_direct_replays',0)+1
        return x

    def query_lower(self,d,x,arrival,state,generator,alternative_generator,packets,*,learn):
        x=self.attach_direct_message(d,x,arrival,state,packets,learn=learn)
        return super().query_lower(d,x,arrival,state,generator,alternative_generator,packets,learn=learn)

    def consume(self,*args,**kwargs):
        self._current_write_vjps=0
        try:
            result=super().consume(*args,**kwargs);state=result[1]
            state['current_write_input_vjps']=state.get('current_write_input_vjps',0)+self._current_write_vjps
            state['direct_producer_edges']={k:e for k,e in state['direct_producer_edges'].items() if k in state['bindings'].entries}
            return result
        finally:
            if hasattr(self,'_current_input_write'):del self._current_input_write
            del self._current_write_vjps

    def input_edges(self,state,key):
        edges=[]
        if key in state['direct_producer_edges']:edges.append(('direct',state['direct_producer_edges'][key]))
        lineage=state['compositional_lineage'].get(key)
        if lineage is not None:edges.extend(('query',r) for r in range(len(lineage.sources)))
        return edges

    def producer_inputs(self,state,key):
        producer=state['producer_records'].get(key);ctx=self._composition_credit_context
        edges=self.input_edges(state,key)
        if producer is None or not ctx['learn'] or not edges:return super().producer_inputs(state,key)
        if ctx['hops']<=0 or ctx['remaining']<=0:raise RuntimeError('recorded depth path cannot be truncated')
        role,p=self.select_credit_role(state,list(range(len(edges))))
        if not 0<p<=1:raise ValueError('positive input-edge inclusion probability required')
        kind,item=edges[role];lineage=state['compositional_lineage'].get(key)
        child,issued=(item.source,item.issued) if kind=='direct' else (lineage.sources[item],lineage.issued[item])
        if child[0]>=key[0]:raise ValueError('input producer DAG must descend in depth')
        ctx['hops']-=1;ctx['remaining']-=1
        try:value=self.read_publication(state,child,issued,ctx['packets'],learn=True)
        finally:ctx['hops']+=1
        base=producer.base_input;views=dict(producer.views)
        if kind=='direct':
            base=base+(value-value.detach())/p;state['direct_producer_replays']+=1
        else:
            aligned=self.aligned_child(key[0],value,lineage,item)
            views[item]=views[item]+(aligned-aligned.detach())/p
            state['memory_transport_heads']+=self.heads
        state['recursive_replay_edges']+=1;state['sampled_credit_paths']+=1
        state['sampled_credit_max_local_inverse_probability']=max(state['sampled_credit_max_local_inverse_probability'],round(1/p))
        stream=base+producer.inbox
        if views:stream=stream+self.assemblers[key[0]-1](views)
        mixed=F.linear(stream,self.mix[key[0]]).reshape(self.heads,self.payload)
        record=state['bindings'].entries[key].native
        return record.residual+(stream-stream.detach()),torch.stack(record.inputs)+(mixed-mixed.detach()),producer

    def storage(self,state):
        result=super().storage(state)
        result['direct_producer_edges']=len(state['direct_producer_edges'])
        result['direct_edge_tensor_bytes']=sum(e.issued.numel()*e.issued.element_size() for e in state['direct_producer_edges'].values())
        return result

def enable_direct_input_credit(model,state):
    if type(model.core) is not SampledCompositionalCredit:raise ValueError('sampled producer-credit parent required')
    before=[id(p) for p in model.parameters()];model.core.__class__=DirectInputProducerCredit
    state.setdefault('direct_producer_edges',{});state.setdefault('direct_producer_replays',0)
    state.setdefault('immediate_direct_replays',0);state.setdefault('current_write_input_vjps',0)
    assert before==[id(p) for p in model.parameters()]
    return dict(new_parameters=0,old_publications='No invented direct lineage; retained unchanged.',scope='One sampled actual input use per descending producer. Direct and query uses counted separately; missing temporal/inbox/token provenance and historical choice credit are outside scope.')
