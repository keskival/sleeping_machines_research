"""Linear-work unbiased content VJP over the available conditional producer DAG.

Sample one actual child per producer and multiply its zero-forward VJP by 1/p.
Covers abstraction depth without an arbitrary two-hop cutoff. Does not recover
unrecorded provenance, historical route credit or exact changed-parameter history.
"""
import torch
from torch.nn import functional as F
from .compositional_producer_credit import CompositionalProducerCredit


class SampledCompositionalCredit(CompositionalProducerCredit):
    def __init__(self,source):
        super().__init__(source);self.replay_hops=self.depth-1;self.replay_edge_budget=self.depth-1

    def new_state(self):
        state=super().new_state();state['composition_credit_rng']=torch.Generator().manual_seed(16919).get_state()
        state['sampled_credit_paths']=0;state['sampled_credit_max_local_inverse_probability']=0
        return state

    def select_credit_role(self,state,roles):
        g=torch.Generator();g.set_state(state['composition_credit_rng'])
        role=roles[int(torch.randint(len(roles),(1,),generator=g))]
        state['composition_credit_rng']=g.get_state()
        return role,1/len(roles)

    def producer_inputs(self,state,key):
        ctx=self._composition_credit_context;lineage=state['compositional_lineage'].get(key);producer=state['producer_records'].get(key)
        if not ctx['learn'] or lineage is None or producer is None:
            return super().producer_inputs(state,key)
        if ctx['hops']<=0 or ctx['remaining']<=0:
            raise RuntimeError('configured path credit cannot truncate a recorded lower-level lineage')
        roles=list(range(len(lineage.sources)));role,p=self.select_credit_role(state,roles);child=lineage.sources[role]
        if child[0]>=key[0]:raise ValueError('producer DAG must descend in level')
        if not 0<p<=1:raise ValueError('positive actual child-inclusion probability required')
        ctx['remaining']-=1;ctx['hops']-=1
        try:value=self.read_publication(state,child,lineage.issued[role],ctx['packets'],learn=True)
        finally:ctx['hops']+=1
        aligned=self.aligned_child(key[0],value,lineage,role);views=dict(producer.views)
        views[role]=views[role]+(aligned-aligned.detach())/p
        state['recursive_replay_edges']+=1;state['memory_transport_heads']+=self.heads;state['sampled_credit_paths']+=1
        state['sampled_credit_max_local_inverse_probability']=max(state['sampled_credit_max_local_inverse_probability'],round(1/p))
        record=state['bindings'].entries[key].native;stream=producer.base_input+producer.inbox+self.assemblers[key[0]-1](views)
        mixed=F.linear(stream,self.mix[key[0]]).reshape(self.heads,self.payload)
        return record.residual+(stream-stream.detach()),torch.stack(record.inputs)+(mixed-mixed.detach()),producer


def enable_sampled_compositional_credit(model,state,seed=16919):
    if type(model.core) is not CompositionalProducerCredit:raise ValueError('compositional producer parent required')
    ids=[id(p) for p in model.parameters()];model.core.__class__=SampledCompositionalCredit
    model.core.replay_hops=model.core.depth-1;model.core.replay_edge_budget=model.core.depth-1
    state.setdefault('composition_credit_rng',torch.Generator().manual_seed(seed).get_state())
    state.setdefault('sampled_credit_paths',0);state.setdefault('sampled_credit_max_local_inverse_probability',0)
    assert ids==[id(p) for p in model.parameters()]
    return dict(new_parameters=0,mode='one actual child per producer, inverse inclusion weighted',max_extra_child_reads_per_root=model.core.depth-1,seed=seed,
        scope='Unbiased content VJP over available frozen-parameter conditional producer DAG in expectation over path sampling. No claim of low variance,unbiased clipped/Adam update,unrecorded temporal/input provenance,historical route credit or exact changed-parameter derivatives.')
