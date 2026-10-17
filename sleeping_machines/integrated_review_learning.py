"""Install reviewed depth/route credit in the complete acquired native learner.

New run only: retains all parameter objects, optimizer references and numerical
state. Old publications receive no invented direct provenance. Episodic writer
snapshots include direct-edge metadata; replay banks exclude live graph contexts.
"""
import sys,types
import torch
from .outcome_compositional_credit import OutcomeCompositionalCredit
from .sampled_compositional_credit import SampledCompositionalCredit
from .deferred_episodic_writer_credit import ProducerCauseBank
from .document_boundary_causes import document_cause_law

class ReviewedProducerCauseBank(ProducerCauseBank):
    def query(self,*args,**kwargs):
        result=super().query(*args,**kwargs)
        if '_writer_credit' in result:
            ctx,snapshot,_=result['_writer_credit']
            snapshot['direct_producer_edges']=dict(ctx['state'].get('direct_producer_edges',{}))
        return result

def enable_reviewed_learning(model,state,*,seed=17371):
    if type(model.core) is not SampledCompositionalCredit or not getattr(model,'_episodic_writer_enabled',False):
        raise ValueError('complete native sampled producer + deferred episodic parent required')
    ids=[id(p) for p in model.parameters()];core=model.core;core.__class__=OutcomeCompositionalCredit
    core._forced_pairs={};core._route_decisions=[];core._replaying=False;core._pending_outcome=None
    for name,value in [('direct_producer_edges',{}),('direct_producer_replays',0),('immediate_direct_replays',0),('current_write_input_vjps',0),('nonlinear_route_audits',0),('nonlinear_route_replay_events',0)]:state.setdefault(name,value)
    state.setdefault('nonlinear_route_rng',torch.Generator().manual_seed(seed).get_state())
    bank=state.get('cause_bank')
    if bank is not None:
        if type(bank) is ProducerCauseBank:bank.__class__=ReviewedProducerCauseBank
        elif not isinstance(bank,ReviewedProducerCauseBank):
            # An unwrapped acquired bank is upgraded on first stream entry.
            if not hasattr(bank,'writer_versions'):bank.writer_versions={}
            bank.__class__=ReviewedProducerCauseBank
    original_stream=model.stream_nll
    def stream(self,inputs,targets,numerical_state,g,a,*,learn=True,stats=False):
        if 'cause_bank' in numerical_state and type(numerical_state['cause_bank']) is ProducerCauseBank:
            numerical_state['cause_bank'].__class__=ReviewedProducerCauseBank
        return original_stream(inputs,targets,numerical_state,g,a,learn=learn,stats=stats)
    model.stream_nll=types.MethodType(stream,model)
    def consume(self,message,numerical_state,g,a,*,learn=True,noise=None,alternative_noise=None,cause_messages=None):
        bank=numerical_state.get('cause_bank')
        if type(bank) is ProducerCauseBank:bank.__class__=ReviewedProducerCauseBank
        if not learn:return self._native_consume(message,numerical_state,g,a,learn=False,noise=noise,alternative_noise=alternative_noise,cause_messages=cause_messages)
        if noise is not None or alternative_noise is not None:raise ValueError('integrated audit uses checkpointed native RNGs')
        if self._pending_outcome is not None:raise RuntimeError('reveal the preceding task outcome before another forecast')
        output,pending=self.consume_for_outcome(message,numerical_state,g,a,cause_messages=cause_messages)
        self._pending_outcome=(pending,numerical_state)
        return output
    core.consume=types.MethodType(consume,core)
    # Existing document/episodic dispatch is preserved for other model owners.
    import experiments.r1_local_native_deep as canonical
    original_law=canonical.indexed_cause_nll
    def law(head,views,target,candidates,**kwargs):
        component,aux,record=original_law(head,views,target,candidates,**kwargs)
        if head is not model.causes or core._pending_outcome is None:return component,aux,record
        pending,numerical_state=core._pending_outcome;core._pending_outcome=None
        prior=torch.log_softmax(model.depth_query(views).squeeze(-1)+model.depth_bias,-1)
        factual_loss=-torch.logsumexp(prior-component,-1)
        token=int(pending.prefix['expected_next_input']) if 'expected_next_input' in pending.prefix else None
        # The actual current input is already appended by the issued stream.
        actual_bank=numerical_state['cause_bank'];token=int(actual_bank.tokens[(actual_bank.count-1)%actual_bank.capacity])
        def task(branch_views,branch,metadata):
            with torch.no_grad():selected,_=model.causes.wake(model.causes.q_unit(branch_views))
            keys=[('receiver',d,h,u) for d,layer in enumerate(metadata['winners']) for h,u in enumerate(layer)]
            keys += [('concept',d,int(u)) for d,row in enumerate(selected) for u in row]
            bank=branch['cause_bank'];stamp=branch['events']-1
            bank.append(token,branch_views,keys,stamp)
            candidate=bank.query(token,keys,stamp,maximum=model.bank_candidates)
            costs,_,_=document_cause_law(model.causes,branch_views,int(target),candidate,explore=False,selected=selected)
            lp=torch.log_softmax(model.depth_query(branch_views).squeeze(-1)+model.depth_bias,-1)
            return -torch.logsumexp(lp-costs,-1)
        audit=torch.Generator().set_state(numerical_state['nonlinear_route_rng'])
        term,rows=core.outcome_credit(pending,factual_loss,task,numerical_state,audit_generator=audit)
        numerical_state['nonlinear_route_rng']=audit.get_state()
        return component,aux+term,record
    canonical.indexed_cause_nll=law
    if 'r1_local_native_deep' in sys.modules:sys.modules['r1_local_native_deep'].indexed_cause_nll=law
    assert ids==[id(p) for p in model.parameters()]
    return dict(new_parameters=0,numerical_reset=False,route_suffix='Actual current-event native suffix and normalized word/cause NLL, one sampled query depth; replayed discovery and numerical writes are audit-only.',scope='Native receiver choices/clocks still have their existing conditional teachers; no whole-history exactness. Token/inbox historical provenance is not reconstructed. All replay/snapshot work must be charged.')
