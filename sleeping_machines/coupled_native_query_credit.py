"""Legal joint credit for two without-replacement native memory queries.

Factual sequential races are retained. One alternative is a complete legal
ordered pair,with the second conditional clock transported under common noise.
The score teacher is exact for the declared linearized joint-assembly objective,
not the full nonlinear future loss or cross-history gradient.
"""
import math
import torch
from .producer_replay_native_race import ProducerReplayNativeRace
from .bound_native_composition import PartBinding
from .sparse_training import _transport


def ordered_pair_probabilities(scores):
    if scores.ndim!=2 or scores.shape[0]!=2 or scores.shape[1]<2:
        raise ValueError('two query score rows and at least two versions required')
    n=scores.shape[1];s=scores.double();mask=torch.eye(n,dtype=torch.bool,device=s.device)
    conditional=s[1].expand(n,n).masked_fill(mask,float('-inf'))
    return (s[0].log_softmax(0)[:,None]+conditional.log_softmax(1)).exp()


class CoupledNativeQueryCredit(ProducerReplayNativeRace):
    def new_state(self):
        state=super().new_state();state.update(joint_pair_score_entries=0,coupled_query_audits=0)
        return state

    def align_replies(self,d,values,times):
        ready=torch.stack(times).max()
        views={r:torch.cat([_transport(v[None],(ready-times[r]).reshape(1),
            self.transport_rate[d,h],self.transport_frequency[d,h])[0]
            for h,v in enumerate(value.split(self.payload))]) for r,value in enumerate(values)}
        return ready,views

    def query_lower(self,d,x,arrival,state,generator,alternative_generator,packets,*,learn):
        candidates=list(state['publications'][d-1])
        if len(candidates)<2:return super().query_lower(d,x,arrival,state,generator,alternative_generator,packets,learn=learn)
        self._producer_depth=d
        keys=torch.stack([state['bindings'].entries[k].native.key_view for k in candidates])
        scores=(self.memory_queries[d-1](x).reshape(2,self.payload)@self.memory_keys[d-1](keys).T/math.sqrt(self.payload)).clamp(-12,12)
        state['memory_key_scores']+=scores.numel();remaining=list(range(len(candidates)))
        chosen=[];indices=[];times=[];raw_times=[];totals=[];parts=[];cache={}
        for role in range(2):
            available=torch.tensor(remaining,dtype=torch.long,device=x.device);rates=scores[role,available].double().exp()
            noise=torch.empty(len(remaining),dtype=torch.float64,device=x.device).exponential_(generator=generator)
            first,local=(noise/rates.detach()).min(0);j=remaining[int(local)];key=candidates[j]
            value=self.read_publication(state,key,arrival,packets,learn=learn)
            state['memory_selected_reads']+=1;state['queried_historical_versions']+=int(key!=candidates[-1])
            log_total=rates.sum().log();clock=first-first*(log_total-log_total.detach())
            times.append(arrival+.001+.010*clock/(1+clock));raw_times.append(first.detach());totals.append(rates.sum().detach())
            chosen.append(value);indices.append(j);cache[j]=value.detach()
            parts.append(PartBinding(key,arrival.detach().clone(),(float(role),0.),'query'));remaining.remove(j)
        ready,views=self.align_replies(d,chosen,times)
        state['memory_transport_heads']+=2*self.heads;state['memory_pair_products']+=self.assemblers[d-1].rank
        # Store only the actual local producer inputs,not counterfactual ones.
        self._producer_inputs[d]=(x.detach().clone(),{r:v.detach().clone() for r,v in views.items()})
        assembled=self.assemblers[d-1](views)
        if learn:
            n=len(candidates);pi=ordered_pair_probabilities(scores).flatten();winner=indices[0]*n+indices[1]
            draws=torch.empty_like(pi).exponential_(generator=alternative_generator)
            alternate_times=(draws/pi.detach()).masked_fill(pi.detach()==0,float('inf'));alternate_times[winner]=float('inf');j=int(alternate_times.argmin());i0,i1=divmod(j,n)
            assert i0!=i1 and j!=winner
            with torch.no_grad():
                values=[]
                for i in [i0,i1]:
                    if i not in cache:
                        cache[i]=self.read_publication(state,candidates[i],arrival,[],learn=False)
                        state['memory_alternative_reads']+=1
                    values.append(cache[i])
                # T1=E1/Lambda1(first): changing the first choice changes the
                # second eligible rate sum. Reuse E1,not an invalid old clock.
                other_rates=scores[1].double().exp();other_total=other_rates[torch.arange(n,device=other_rates.device)!=i0].sum()
                t1=raw_times[1]*totals[1]/other_total
                alternative_reply_times=[arrival.detach()+.001+.010*t/(1+t) for t in [raw_times[0],t1]]
                _,alternate_views=self.align_replies(d,values,alternative_reply_times)
                alternative=self.assemblers[d-1](alternate_views)
                state['memory_transport_heads']+=2*self.heads;state['memory_pair_products']+=self.assemblers[d-1].rank
            q=(pi[j]/(1-pi[winner])).detach();coefficient=(pi[j]-pi[j].detach())/q
            assembled=assembled+coefficient.to(assembled.dtype)*(alternative-assembled).detach()
            state['joint_pair_score_entries']+=n*n;state['coupled_query_audits']+=1
        return x+assembled,ready,tuple(parts)


def enable_coupled_query_credit(model,state):
    if type(model.core) is not ProducerReplayNativeRace:raise ValueError('producer-credit parent required')
    identities=[id(p) for p in model.parameters()];model.core.__class__=CoupledNativeQueryCredit
    state.setdefault('joint_pair_score_entries',0);state.setdefault('coupled_query_audits',0)
    assert identities==[id(p) for p in model.parameters()]
    return dict(new_parameters=0,factual_races='Same sequential without-replacement draws and times.',
        teacher='Complete legal ordered-pair alternative with correct conditional second clock;conditional linearized joint-assembly teacher,not exact full-sequence credit.')
