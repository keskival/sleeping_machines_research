"""Bounded local producer credit for historical native memory reads.

Retains numerical local inputs/prior selected state,not an event graph. Historical
reads replay only the local joint composer/mix and last native input-write map.
Own-write-parameter eligibility remains separate;write parameters are detached
in the input VJP to avoid double credit. No cross-part recursive replay/BPTT.
"""
from dataclasses import dataclass
import math
import torch
from torch.nn import functional as F
from .packed_queryable_local_deep_race import PackedQueryableLocalDeepRace,HistoricalReadCredit


@dataclass(frozen=True)
class ProducerRecord:
    base_input: torch.Tensor
    views: dict
    inbox: torch.Tensor
    prior_value: torch.Tensor
    write_parameters: dict
    elapsed: torch.Tensor


def local_write(prior,params,incoming,elapsed):
    normal=F.layer_norm(incoming,(incoming.shape[-1],))
    control=torch.einsum('hcp,hp->hc',params['control_w'],normal)+params['control_b']
    forget=F.softplus(control[:,0])/math.log(2);write=2*control[:,1].sigmoid()
    rate=F.softplus(params['raw_rate'])+1e-6
    magnitude=torch.exp(-elapsed.to(incoming.dtype)[:,None]*rate*forget[:,None])
    angle=elapsed[:,None]*params['frequency'].double()
    rotated=torch.complex(magnitude*angle.cos().to(incoming.dtype),magnitude*angle.sin().to(incoming.dtype))*prior
    b=torch.einsum('hpq,hq->hp',params['input'],incoming).reshape(incoming.shape[0],-1,2)
    value=rotated+write[:,None]*torch.complex(b[...,0],b[...,1])
    return torch.view_as_real(value).flatten(-2)


class ProducerReplayNativeRace(PackedQueryableLocalDeepRace):
    def new_state(self):
        state=super().new_state();state['producer_records']={};state['producer_replay_reads']=0
        return state

    def query_lower(self,d,x,arrival,state,generator,alternative_generator,packets,*,learn):
        self._producer_depth=d;captured={}
        def capture(module,args):captured.update({r:v.detach().clone() for r,v in args[0].items()})
        handle=self.assemblers[d-1].register_forward_pre_hook(capture)
        try:result=super().query_lower(d,x,arrival,state,generator,alternative_generator,packets,learn=learn)
        finally:handle.remove()
        self._producer_inputs[d]=(x.detach().clone(),captured)
        return result

    def proposal(self,old,params,incoming,elapsed,*,traces):
        if traces:
            keys=['control_w','control_b','input','raw_rate','frequency']
            self._producer_writes[self._producer_depth]=(old['value'].detach().clone(),
                {k:params[k].detach().clone() for k in keys},elapsed.detach().clone())
        return super().proposal(old,params,incoming,elapsed,traces=traces)

    def publish_layer(self,d,state,arrival,head_times,winners,parts,mixed,stream,x):
        super().publish_layer(d,state,arrival,head_times,winners,parts,mixed,stream,x)
        key=state['publications'][d][-1];prior,params,elapsed=self._producer_writes[d]
        base,views=self._producer_inputs[d]
        inbox=self._producer_inboxes[d]
        state['producer_records'][key]=ProducerRecord(base,views,inbox,prior,params,elapsed)

    def producer_inputs(self,state,key):
        entry=state['bindings'].entries[key];record=entry.native;producer=state['producer_records'].get(key)
        saved=torch.stack(record.inputs)
        # Migrated pre-extension publications have no invented producer record.
        if producer is None:return record.residual,saved,None
        d=key[0];stream=producer.base_input+producer.inbox
        if producer.views:stream=stream+self.assemblers[d-1](producer.views)
        mixed=F.linear(stream,self.mix[d]).reshape(self.heads,self.payload)
        return (record.residual+(stream-stream.detach()),
            saved+(mixed-mixed.detach()),producer)

    def read_publication(self,state,key,now,packets,*,learn=True):
        entry=state['bindings'].entries[key];record=entry.native
        stamp=torch.as_tensor(now,dtype=torch.float64,device=record.key_view.device)
        if not bool(torch.isfinite(stamp)) or float(stamp.detach())<float(entry.available):
            raise ValueError('lower publication must be available before query')
        d=key[0];hi=torch.arange(self.heads,device=record.key_view.device)
        winners=torch.tensor(record.winners,dtype=torch.long,device=hi.device)
        params={name:bank[d,hi,winners] for name,bank in self.banks.items()}
        leaf=torch.stack([t.memory() for t in record.receivers]).requires_grad_(learn)
        if learn:packets.append(HistoricalReadCredit(d,record.receivers,leaf,record.winners))
        residual,incoming,producer=self.producer_inputs(state,key)
        memory=leaf
        if learn and producer is not None:
            replay=local_write(producer.prior_value,producer.write_parameters,incoming,producer.elapsed)
            memory=leaf+(replay-replay.detach());state['producer_replay_reads']+=1
        values=self.unit_value(params,incoming,memory)
        age=stamp-record.head_times
        magnitude=torch.exp(-age.to(values.dtype)[:,None]*(F.softplus(self.transport_rate[d])+1e-6))
        angle=age[:,None]*self.transport_frequency[d].double();pairs=values.reshape(self.heads,-1,2)
        real,imag=pairs[...,0]*magnitude,pairs[...,1]*magnitude;c,s=angle.cos().to(values.dtype),angle.sin().to(values.dtype)
        aligned=torch.stack((c*real-s*imag,s*real+c*imag),-1).flatten()
        state['bindings'].reads+=1;state['memory_transport_heads']+=self.heads
        return F.layer_norm(residual+aligned/math.sqrt(self.depth),(self.width,))

    def consume(self,message,state,generator,alternative_generator,**kwargs):
        self._producer_depth=0;self._producer_inputs={0:(message.detach().clone(),{})};self._producer_writes={}
        inboxes=kwargs.get('cause_messages')
        self._producer_inboxes=[message.new_zeros(self.width) if inboxes is None else inboxes[d].detach().clone() for d in range(self.depth)]
        try:
            result=super().consume(message,state,generator,alternative_generator,**kwargs)
            live=state['bindings'].entries
            state['producer_records']={k:v for k,v in state['producer_records'].items() if k in live}
            return result
        finally:
            del self._producer_inputs,self._producer_writes,self._producer_inboxes,self._producer_depth

    def storage(self,state):
        result=super().storage(state)
        result.update(producer_records=len(state['producer_records']),producer_tensor_bytes=sum(
            sum(v.numel()*v.element_size() for v in [r.base_input,r.inbox,r.prior_value,r.elapsed,*r.views.values(),*r.write_parameters.values()])
            for r in state['producer_records'].values()),scope=result['scope']+' Additional bounded local producer inputs/frozen write coefficients;current replay graphs/packets are paid. No whole-history replay.')
        return result


def enable_producer_credit(model,state):
    """Explicit schema/method upgrade; parameter objects and Adam refs survive."""
    if type(model.core) is not PackedQueryableLocalDeepRace:
        raise ValueError('expected the admitted packed queryable native core')
    identities=[id(p) for p in model.parameters()]
    model.core.__class__=ProducerReplayNativeRace
    state.setdefault('producer_records',{});state.setdefault('producer_replay_reads',0)
    assert identities==[id(p) for p in model.parameters()]
    return dict(new_parameters=0,parameter_objects_preserved=True,
        old_publications='Preserved;no invented producer inputs. New publications carry bounded local producer records.',
        credit='Historical reads now teach the local composer/mix and last input-write path;original own-write eligibility still supplies native write-parameter credit. No recursive replay through earlier child producers.')
