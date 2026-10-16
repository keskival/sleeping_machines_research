"""Identity-preserving native role composition and issued-version teaching.

Participants are addressed memories, never geometric-role slots. All current
participants/pairs are charged: sparse partner discovery is not implemented.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .part_identity_composition import PartIdentityComposition,shape_signature
from .bound_native_composition import BoundTimerObject
from .relational_frame_composition import RelationalEventCore


class IdentityRoleCore(RelationalEventCore):
    def __init__(self,previous):
        nn.Module.__init__(self)
        self.base,self.composer,self.width=previous.base,previous.composer,previous.width
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1601)
            self.same_role_weight=nn.Parameter(torch.randn(self.width,4,self.composer.rank)*(.15/math.sqrt(self.composer.rank)))
            self.relative_weight=nn.Parameter(torch.randn(self.width,2,self.composer.rank)*(.15/math.sqrt(self.composer.rank)))

    def new_state(self):
        state=super().new_state();state.participants={};state.current_part=None
        state.identity_relation_pairs=state.identity_relation_reads=state.identity_relation_madds=0
        return state

    def relation(self,entries):
        # entries: immutable source key, transported content, observed pose.
        # Address ordering only enumerates pairs; symmetric+antisymmetric terms
        # below make the result invariant to enumeration and address relabeling.
        if not entries:return self.same_role_weight.new_zeros(self.width)
        c=self.composer;left=[];right=[];roles=[];poses=[]
        for _,value,pose in entries:
            v=value/torch.sqrt(value.square().mean()+1e-6)
            left.append(F.linear(v,c.left));right.append(F.linear(v,c.right))
            roles.append(2*int(pose[0]>=.5)+int(pose[1]>=.5));poses.append(value.new_tensor(pose))
        out=torch.stack([F.linear(v,c.role_weight[:,role]) for v,role in zip(left,roles)]).sum(0)/math.sqrt(len(entries))
        pairs=[]
        for i in range(len(entries)):
            for j in range(i):
                a,b=sorted((roles[i],roles[j]));w=self.same_role_weight[:,a] if a==b else c.pair_weight[:,c.pairs.index((a,b))]
                symmetric=(left[i]*right[j]+left[j]*right[i])*.5
                antisymmetric=(left[i]*right[j]-left[j]*right[i])*.5
                delta=poses[i]-poses[j]
                term=F.linear(symmetric,w)
                for axis in range(2):term=term+F.linear(antisymmetric*torch.tanh(delta[axis]),self.relative_weight[:,axis])
                pairs.append(term)
        if pairs:out=out+torch.stack(pairs).sum(0)/math.sqrt(len(pairs))
        return out

    def consume_event(self,source,timestamp,content,state):
        mark=torch.as_tensor(content,dtype=self.base.content.weight.dtype,device=self.base.content.weight.device)
        stamp=torch.as_tensor(timestamp,dtype=torch.float64,device=mark.device)
        if mark.shape!=(self.base.content_dim,) or not bool(torch.isfinite(mark).all()) or not bool(torch.isfinite(stamp)) or float(stamp.detach())<state.last_input_time or not 0<=int(source)<self.base.sources:
            raise ValueError('known source,finite native content and causal time required')
        if float(mark[self.width+2].detach())>=0:
            if state.current_part is None:raise ValueError('native relation requires an issued child binding')
            binding=state.current_part;group=float(mark[self.width+2].detach())
            state.participants[binding.source[:2]]=(group,mark[:self.width],stamp,binding.pose,binding.source)
            read_time=stamp if source not in state.contexts else torch.maximum(stamp,state.contexts[source][1].max())
            entries=[]
            for key,(g,value,time,pose,version) in sorted(state.participants.items()):
                if g!=group:continue
                aligned=torch.cat([self.base.transport(v,read_time-time,self.base.depth-1,h) for h,v in enumerate(value.split(self.base.payload))])
                entries.append((key,aligned,pose))
            relation=self.relation(entries);mark=torch.cat((mark[:self.width]+relation,mark[self.width:]))
            k=len(entries);pairs=k*(k-1)//2;rank=self.composer.rank
            state.identity_relation_reads+=k;state.identity_relation_pairs+=pairs
            state.identity_relation_madds+=(3*k+3*pairs)*rank*self.width
            state.role_transport_calls+=k*self.base.heads
        return self.base.consume_event(source,timestamp,mark,state)


class IdentityRoleObject(BoundTimerObject):
    def accept(self,state,address,timestamp,content,event_identity):
        binding=state.binding_store.inflight[(state.binding_level,address,id(content))]
        # IndexedTemporalObject creates the private state lazily. Validate and
        # let the native duplicate check decide whether a notification is new.
        if address not in state.private:state.private[address]=self.core.new_state()
        private=state.private[address];previous=private.current_part;private.current_part=binding
        try:return super().accept(state,address,timestamp,content,event_identity)
        finally:private.current_part=previous


class IdentityRoleComposition(PartIdentityComposition):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        for level in range(1,self.levels):
            old=self.objects[level];self.objects[level]=IdentityRoleObject(IdentityRoleCore(old.core),old.width)

    def load_identity_actor(self,weights):
        result=self.load_state_dict(weights,strict=False)
        if result.unexpected_keys or any(not (k.endswith('.same_role_weight') or k.endswith('.relative_weight')) for k in result.missing_keys):
            raise ValueError('identity actor migration changed existing coverage')

    def identity_loss(self,state,events):
        rows=torch.as_tensor(events,dtype=torch.float64)
        if state.queue or state.ownership is None or len(rows)!=sum(state.input_issued.values()) or float(rows[:,0].max())>state.now:
            raise ValueError('teacher requires exactly the drained revealed scene')
        labels={}
        for address in {a for a,_ in state.ownership.values()}:
            pixels={p for p,(a,_) in state.ownership.items() if a==address}
            sig=shape_signature(rows[[tuple(map(int,p)) in pixels for p in rows[:,1:3]]])
            if sig not in self.shape_vocabulary:raise ValueError('unknown observed TRAIN shape')
            labels[address]=self.shape_vocabulary[sig]
        # Supervise actual level-one views retained by parents at issue time,
        # rather than a later read of the changing lower memory. Deduplicate
        # identical versions/times shared by different parents.
        used={}
        for entry in state.bindings.entries.values():
            for part in entry.parts:
                if part.source[0]==1:used[(part.source,float(part.issued.detach()))]=part
        losses=[];correct=0
        for _,part in sorted(used.items()):
            value=self.read_bound(state,part.source,part.issued)
            logits=self.identity_head(value);label=labels[part.source[1]]
            losses.append(F.cross_entropy(logits[None],torch.tensor([label],device=logits.device)))
            correct+=int(logits.detach().argmax()==label)
        if not losses:raise ValueError('no parent-used part versions')
        state.identity_work=dict(identity_reads=len(losses),identity_head_madds=len(losses)*self.width*len(labels),identity_target_event_tests=len(labels)*len(rows))
        return torch.stack(losses).mean(),dict(correct=correct,parts=len(losses),endpoint='parent-used immutable versions at issue time')

    def work(self,state):
        result=super().work(state);private=[p for b in state.banks[1:] for p in b.private.values()]
        result.update(identity_relation_reads=sum(p.identity_relation_reads for p in private),identity_relation_pairs=sum(p.identity_relation_pairs for p in private),identity_relation_madds=sum(p.identity_relation_madds for p in private),identity_role_slots=sum(len(p.participants) for p in private),identity_role_tensor_bytes=sum(v.numel()*v.element_size()+t.numel()*t.element_size() for p in private for _,v,t,_,_ in p.participants.values()))
        result['identity_relation_scope']='All current addressed child views and quadratic pairs;native receipt/timer races retained. No sparse partner selection or full arithmetic/energy claim.'
        return result

    def checkpoint_scene(self,state):
        """Inference/numerical-state checkpoint; does not serialize credit graphs.

        Stable queue serials replace Python tensor IDs across torch.save/load.
        Training resume uses model/optimizer/RNG at scene boundaries instead.
        """
        envelopes={}
        for _,serial,_,kind,data in state.queue:
            if kind=='deliver':
                level,address,content,_=data
                envelopes[serial]=state.bindings.inflight[(level,address,id(content))]
        if len(envelopes)!=len(state.bindings.inflight):raise ValueError('unmatched in-flight binding')
        return dict(state=state,envelopes=envelopes)

    def restore_scene(self,checkpoint):
        state=checkpoint['state'];envelopes=checkpoint['envelopes'];inflight={}
        for _,serial,_,kind,data in state.queue:
            if kind=='deliver':
                level,address,content,_=data
                inflight[(level,address,id(content))]=envelopes[serial]
        if len(inflight)!=len(envelopes):raise ValueError('unmatched checkpoint envelope')
        state.bindings.inflight=inflight
        return state
