"""R1 fully native deep race prototype, with no recurrent BPTT.

Reuses CERL's exactly normalized cause laws with ZERO temporal-memory/MLP
backbone layers. All state comes from LocalDeepRace. The depth axis is itself
a competing-cause mixture, giving every layer a direct local task cotangent.
Current-event graphs and bounded episodic write graphs are still charged;
this is not yet an asynchronous optimizer or exact whole-network RTRL.
"""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'experiments')]
import torch
from torch import nn
from r1_cerl import CERL,word_classes,work_per_target
from sleeping_machines.local_deep_race import LocalDeepRace


class NativeEventCauseHead(CERL):
    def states(self,x):
        if not hasattr(self,'native_views'):
            raise ValueError('native event views required;no alternative backbone exists')
        return self.native_views


class LocalNativeDeepLM(nn.Module):
    def __init__(self,vocabulary,class_of,class_words,width=64,depth=8,heads=4,pool=16,
                 concept_units=4096,topk=4,explore=2,dk=32):
        super().__init__()
        self.width,self.depth=width,depth
        self.core=LocalDeepRace(width,depth,heads,pool)
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
        features=[];packets=[]
        for token in inputs:
            h,state,local,_=self.core.consume(self.causes.embed(token),state,generator,alternative_generator,learn=learn)
            features.append(h);packets.extend(local)
        h=torch.stack(features)[None]  # [1,time,depth,width]
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
                    learning_scope='Receiver-local own-write traces,local routing surrogate,layer-local message boundaries. No recurrent BPTT. Current-event and bounded episodic-write graphs retained within this credit batch. Changing-parameter history and cross-producer full gradients are not exact.')
        return nll,objective,state,packets,report

    def deliver_credit(self,packets,state):self.core.deliver_credit(packets,state)
