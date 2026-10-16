"""R1 fully native deep race prototype, with no recurrent BPTT.

Reuses CERL's exactly normalized cause laws with ZERO temporal-memory/MLP
backbone layers. All state comes from LocalDeepRace. The depth axis is itself
a competing-cause mixture, giving every layer a direct local task cotangent.
Current-event graphs and bounded episodic write graphs are still charged;
this is not yet an asynchronous optimizer or exact whole-network RTRL.
"""
import sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'experiments')]
import torch
from torch import nn
from r1_cerl import CERL,word_classes,work_per_target
from sleeping_machines.local_deep_race import LocalDeepRace
from sleeping_machines.packed_local_deep_race import PackedLocalDeepRace
from sleeping_machines.native_cause_bank import NativeCauseBank
from sleeping_machines.native_indexed_cause_head import indexed_cause_nll


class NativeEventCauseHead(CERL):
    def states(self,x,msg=None):
        if msg is not None:
            raise ValueError('cause messages must enter native receiver events,not a hidden carrier')
        if not hasattr(self,'native_views'):
            raise ValueError('native event views required;no alternative backbone exists')
        return self.native_views


class LocalNativeDeepLM(nn.Module):
    def __init__(self,vocabulary,class_of,class_words,width=64,depth=8,heads=4,pool=16,
                 concept_units=4096,topk=4,explore=2,dk=32,packed=False,bank_capacity=32768,bank_candidates=128):
        super().__init__()
        self.width,self.depth=width,depth
        self.bank_capacity,self.bank_candidates=bank_capacity,bank_candidates
        self.core=LocalDeepRace(width,depth,heads,pool)
        if packed:self.core=PackedLocalDeepRace(self.core)
        # layers=0 constructs no SSM/attention/MLP carrier. Only the native
        # cause distributions,word law and token embedding are reused.
        self.causes=NativeEventCauseHead(vocabulary,class_of,class_words,d=width,layers=0,
            dk=dk,units=concept_units,topk=topk,explore=explore,product_keys=True)
        assert len(self.causes.tm)==len(self.causes.mlp)==0
        self.depth_query=nn.Linear(width,1,bias=False);nn.init.zeros_(self.depth_query.weight)
        self.depth_bias=nn.Parameter(torch.zeros(depth))

    def new_state(self):return self.core.new_state()

    def nll(self,inputs,targets,state,generator,alternative_generator,*,learn=True,stats=False):
        if inputs.ndim!=1 or targets.shape!=inputs.shape or inputs.dtype!=torch.long or not len(inputs):
            raise ValueError('one nonempty contiguous token lane required')
        features=[];packets=[];started=time.perf_counter()
        for token in inputs:
            h,state,local,_=self.core.consume(self.causes.embed(token),state,generator,alternative_generator,learn=learn)
            features.append(h);packets.extend(local)
        h=torch.stack(features)[None]  # [1,time,depth,width]
        native_forward_s=time.perf_counter()-started;head_started=time.perf_counter()
        losses=[];aux=[];reports=[];background=[]
        for d in range(self.depth):
            self.causes.native_views=h[:,:,d]
            try:
                result=self.causes.lane_nll(inputs[None],targets[None],stats=stats)
                if stats:
                    loss,report=result;reports.append(report)
                    background.append(-self.causes.two_level(h[:,:,d].reshape(-1,1,self.width),targets).reshape(1,-1))
                else:loss=result
                losses.append(loss)
                if self.causes.aux is not None:aux.append(self.causes.aux.mean())
            finally:del self.causes.native_views
        component_nll=torch.stack(losses,-1)
        log_prior=torch.log_softmax(self.depth_query(h).squeeze(-1)+self.depth_bias,-1)
        nll=-torch.logsumexp(log_prior-component_nll,-1)
        objective=nll.mean()+(torch.stack(aux).mean() if aux else 0.)
        report={}
        if stats:
            with torch.no_grad():
                bg_nll=-torch.logsumexp(log_prior-torch.stack(background,-1),-1)
                report=dict(nll=float(nll.mean()),background_only_nll=float(bg_nll.mean()),
                    native_cause_gain=float((bg_nll-nll).mean()),
                    depth_responsibility=torch.softmax(log_prior-component_nll,-1).mean((0,1)).tolist(),
                    cause_reports_by_depth=reports,
                    work_per_depth_at_history=work_per_target(self.causes,len(inputs)),
                    native_counts={k:v for k,v in state.items() if isinstance(v,int)},
                    persistent_storage=self.core.storage(state),
                    native_forward_s=native_forward_s,cause_head_forward_s=time.perf_counter()-head_started,
                    learning_scope='Receiver-local own-write traces,local routing surrogate,layer-local message boundaries. No recurrent BPTT. Current-event and bounded episodic-write graphs retained within this credit batch. Changing-parameter history and cross-producer full gradients are not exact.')
        return nll,objective,state,packets,report

    def deliver_credit(self,packets,state):self.core.deliver_credit(packets,state)

    def stream_nll(self,inputs,targets,state,generator,alternative_generator,*,learn=True,stats=False):
        """Persistent indexed causes and causal cause messages across credit batches.

        Each target is scored before it produces the next event's message.
        Stored event values and inbox messages are numerical,not temporal graphs.
        Only local receiver eligibility carries own-write parameter credit.
        """
        if inputs.ndim!=1 or targets.shape!=inputs.shape or not len(inputs) or inputs.dtype!=torch.long:
            raise ValueError('one nonempty contiguous token stream required')
        if not torch.equal(inputs[1:],targets[:-1]):raise ValueError('causally shifted targets required')
        if 'expected_next_input' in state and int(inputs[0])!=state['expected_next_input']:
            raise ValueError('numerical history may only continue on its actual next token')
        if 'cause_bank' not in state:
            state['cause_bank']=NativeCauseBank(self.depth,self.width,self.bank_capacity)
            state['cause_inbox']=self.causes.embed.weight.new_zeros(self.depth,self.width)
        bank=state['cause_bank'];losses=[];auxiliaries=[];packets=[];reports=[]
        started=time.perf_counter()
        for token,target in zip(inputs,targets):
            h,state,local,metadata=self.core.consume(self.causes.embed(token),state,generator,alternative_generator,
                learn=learn,cause_messages=state['cause_inbox'])
            packets.extend(local)
            with torch.no_grad():
                selected,_=self.causes.wake(self.causes.q_unit(h.detach()))
                keys=[('receiver',d,head,unit) for d,layer in enumerate(metadata['winners']) for head,unit in enumerate(layer)]
                keys += [('concept',d,int(unit)) for d,row in enumerate(selected) for unit in row]
            stamp=state['events']-1
            bank.append(int(token),h,keys,stamp)
            candidates=bank.query(int(token),keys,stamp,maximum=self.bank_candidates)
            component,aux,record=indexed_cause_nll(self.causes,h,int(target),candidates,
                explore=learn and self.training,selected=selected)
            log_prior=torch.log_softmax(self.depth_query(h).squeeze(-1)+self.depth_bias,-1)
            nll=-torch.logsumexp(log_prior-component,-1);losses.append(nll);auxiliaries.append(aux)
            with torch.no_grad():
                posterior=torch.softmax(log_prior-component,-1)
                # Attribution matures only after the token is revealed. The
                # next receiver event receives the message without a query.
                state['cause_inbox']=posterior[:,None]*record['posterior_message']
                if stats:
                    background=-torch.logsumexp(log_prior-record['background_nll'],-1)
                    reports.append(dict(nll=float(nll),background=float(background),depth=posterior,
                        cause=(posterior[:,None]*record['cause_responsibility']).sum(0),candidates=record['candidate_events']))
        state['expected_next_input']=int(targets[-1])
        nll=torch.stack(losses);objective=nll.mean()+torch.stack(auxiliaries).mean();report={}
        if stats:
            report=dict(nll=float(nll.detach().mean()),background_only_nll=sum(r['background'] for r in reports)/len(reports),
                native_cause_gain=sum(r['background']-r['nll'] for r in reports)/len(reports),
                depth_responsibility=torch.stack([r['depth'] for r in reports]).mean(0).tolist(),
                cause_responsibility=torch.stack([r['cause'] for r in reports]).mean(0).tolist(),
                delivered_candidates=sum(r['candidates'] for r in reports),bank=bank.storage(),
                persistent_storage=self.core.storage(state),forward_s=time.perf_counter()-started,
                learning_scope='Persistent indexed episodic causes and outcome-matured messages into native receivers. No recurrent BPTT or old episodic writer graph;read-time keys and receiver-local own-write eligibility learn. Parameter-changing history is approximate;optimizer/versioning separate.')
        return nll,objective,state,packets,report
