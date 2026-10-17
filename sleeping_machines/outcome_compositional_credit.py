"""Actual nonlinear current-event suffix teaching for legal joint queries.

Issued forecasts are unchanged. After outcome revelation, replay one legal
alternative from the same numerical prefix and common unit-hazard clocks.
This estimates categorical query-choice credit for a bounded one-event suffix,
not whole-history policy/timing credit or every native receiver choice.
"""
import copy,math
from dataclasses import dataclass
import torch
from .direct_input_producer_credit import DirectInputProducerCredit
from .compositional_producer_credit import CompositionalLineage
from .bound_native_composition import PartBinding
from .coupled_native_query_credit import ordered_pair_probabilities
from .composition_task_credit import actual_joint_loss_credit

@dataclass
class QueryOutcomeDecision:
    depth:int
    scores:torch.Tensor
    factual:tuple
    alternative:tuple
    hazards:tuple
    proposal:float

@dataclass
class PendingQueryOutcome:
    prefix:dict
    message:torch.Tensor
    inbox:object
    g:torch.Tensor
    alt:torch.Tensor
    decisions:list
    versions:tuple
    delivered:bool=False

class OutcomeCompositionalCredit(DirectInputProducerCredit):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self._forced_pairs={};self._route_decisions=[];self._replaying=False

    def new_state(self):
        state=super().new_state();state.update(nonlinear_route_audits=0,nonlinear_route_replay_events=0)
        return state

    def query_lower(self,d,x,arrival,state,generator,alternative_generator,packets,*,learn):
        candidates=list(state['publications'][d-1])
        if len(candidates)<2:return super().query_lower(d,x,arrival,state,generator,alternative_generator,packets,learn=learn)
        x=self.attach_direct_message(d,x,arrival,state,packets,learn=learn)
        self._producer_depth=d
        keys=torch.stack([state['bindings'].entries[k].native.key_view for k in candidates])
        scores=(self.memory_queries[d-1](x).reshape(2,self.payload)@self.memory_keys[d-1](keys).T/math.sqrt(self.payload)).clamp(-12,12)
        state['memory_key_scores']+=scores.numel();remaining=list(range(len(candidates)))
        chosen=[];indices=[];times=[];hazards=[];parts=[];forced=self._forced_pairs.get(d)
        for role in range(2):
            available=torch.tensor(remaining,dtype=torch.long,device=x.device);rates=scores[role,available].double().exp()
            noise=torch.empty(len(remaining),dtype=torch.float64,device=x.device).exponential_(generator=generator)
            first,local=(noise/rates.detach()).min(0);j=remaining[int(local)]
            if forced is not None:
                j=forced[0][role]
                if j not in remaining:raise ValueError('forced pair must be legal in its original prefix')
                first=forced[1][role].to(rates)/rates.sum().detach()
            key=candidates[j];value=self.read_publication(state,key,arrival,packets,learn=learn)
            state['memory_selected_reads']+=1;state['queried_historical_versions']+=int(key!=candidates[-1])
            total=rates.sum();clock=first-first*(total.log()-total.log().detach())
            times.append(arrival+.001+.010*clock/(1+clock));hazards.append((first*total).detach())
            chosen.append(value);indices.append(j);parts.append(PartBinding(key,arrival.detach().clone(),(float(role),0.),'query'));remaining.remove(j)
        ready,views=self.align_replies(d,chosen,times)
        state['memory_transport_heads']+=2*self.heads;state['memory_pair_products']+=self.assemblers[d-1].rank
        self._producer_inputs[d]=(x.detach().clone(),{r:v.detach().clone() for r,v in views.items()})
        self._lineage_inputs[d]=CompositionalLineage(tuple(p.source for p in parts),tuple(p.issued.detach().clone() for p in parts),tuple(t.detach().clone() for t in times),ready.detach().clone())
        assembled=self.assemblers[d-1](views)
        if learn and not self._replaying:
            n=len(candidates);pi=ordered_pair_probabilities(scores).flatten();factual=indices[0]*n+indices[1]
            weights=pi.detach().clone();weights[factual]=0.;mass=weights.sum()
            if not bool(torch.isfinite(mass)) or float(mass)<=0:raise RuntimeError('nonfactual legal support required')
            draws=torch.empty_like(pi).exponential_(generator=alternative_generator)
            alt=int((draws/weights).masked_fill(weights==0,float('inf')).argmin())
            self._route_decisions.append(QueryOutcomeDecision(d,scores,tuple(indices),divmod(alt,n),tuple(hazards),float(weights[alt]/mass)))
            state['joint_pair_score_entries']+=n*n
        return x+assembled,ready,tuple(parts)

    def _native_consume(self,*args,**kwargs):
        return super().consume(*args,**kwargs)

    @staticmethod
    def numerical_prefix(state):
        source=dict(state)
        bank=source.get('cause_bank')
        if bank is not None and hasattr(bank,'_writer_context'):
            bank=copy.copy(bank);del bank._writer_context;source['cause_bank']=bank
        return copy.deepcopy(source)

    def consume_for_outcome(self,message,state,generator,alternative_generator,*,cause_messages=None):
        prefix=self.numerical_prefix(state);gs=generator.get_state();als=alternative_generator.get_state()
        self._route_decisions=[]
        output=self._native_consume(message,state,generator,alternative_generator,learn=True,cause_messages=cause_messages)
        pending=PendingQueryOutcome(prefix,message.detach().clone(),None if cause_messages is None else cause_messages.detach().clone(),gs,als,list(self._route_decisions),tuple(p._version for p in self.parameters()))
        return output,pending

    def outcome_credit(self,pending,factual_loss,loss_from_views,state,*,audit_generator=None):
        if pending.delivered:raise RuntimeError('query outcome credit must be delivered once')
        if pending.versions!=tuple(p._version for p in self.parameters()):raise RuntimeError('parameters changed before deferred query teaching')
        factual_loss=torch.as_tensor(factual_loss).detach()
        if factual_loss.ndim or not bool(torch.isfinite(factual_loss)):raise ValueError('finite revealed factual task loss required')
        decisions=pending.decisions
        if not decisions:
            pending.delivered=True;return factual_loss.new_zeros(()),[]
        if audit_generator is None:selected=decisions;multiplier=1.
        else:
            selected=[decisions[int(torch.randint(len(decisions),(1,),generator=audit_generator))]];multiplier=float(len(decisions))
        result=factual_loss.new_zeros(());rows=[];saved_rng=torch.get_rng_state();saved_decisions=self._route_decisions
        try:
            self._replaying=True
            for decision in selected:
                self._forced_pairs={decision.depth:(decision.alternative,decision.hazards)}
                branch=copy.deepcopy(pending.prefix);g=torch.Generator().set_state(pending.g);a=torch.Generator().set_state(pending.alt)
                with torch.no_grad():
                    views,branch,_,metadata=self._native_consume(pending.message,branch,g,a,learn=False,cause_messages=pending.inbox)
                    alternative_loss=loss_from_views(views,branch,metadata)
                term=actual_joint_loss_credit(decision.scores,decision.factual,decision.alternative,factual_loss,alternative_loss,decision.proposal)
                result=result+multiplier*term.to(result.dtype)
                state['nonlinear_route_audits']+=1;state['nonlinear_route_replay_events']+=1
                rows.append(dict(depth=decision.depth,factual_loss=float(factual_loss),alternative_loss=float(alternative_loss),proposal=decision.proposal,depth_inverse_probability=multiplier))
        finally:
            self._forced_pairs={};self._replaying=False;self._route_decisions=saved_decisions;torch.set_rng_state(saved_rng)
        pending.delivered=True
        return result,rows
