"""Native temporal parent memories with persistent child-role interactions.

Known role-conditioned/bilinear primitives support the native learner. They do
not replace native temporal races, selected state writes or emitted views.
"""
import itertools,math
import torch
from torch import nn
from torch.nn import functional as F
from .frame_native_composition import FrameNativeComposition


class ChildRoleComposer(nn.Module):
    pairs=tuple(itertools.combinations(range(4),2))
    def __init__(self,width,rank=None,seed=601):
        super().__init__();self.width=width;self.rank=rank or width
        if not 0<self.rank<=width:raise ValueError('rank must retain at least one channel')
        left=torch.zeros(self.rank,width);right=torch.zeros_like(left)
        for i in range(self.rank):left[i,i]=1.;right[i,(i+1)%width]=1.
        self.left=nn.Parameter(left);self.right=nn.Parameter(right)
        # Nonzero relational operators at every level: no stack of zero gates.
        # Keep native constructor/race RNG untouched by auxiliary initialization.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.role_weight=nn.Parameter(torch.randn(width,4,self.rank)*(.15/math.sqrt(self.rank)))
            self.pair_weight=nn.Parameter(torch.randn(width,6,self.rank)*(.15/math.sqrt(self.rank)))

    def forward(self,views):
        if not views:return self.left.new_zeros(self.width)
        if any(r not in range(4) for r in views):raise ValueError('four geometric child roles required')
        roles=sorted(views)
        normalized={r:views[r]/torch.sqrt(views[r].square().mean()+1e-6) for r in roles}
        left={r:F.linear(normalized[r],self.left) for r in roles}
        right={r:F.linear(normalized[r],self.right) for r in roles[1:]}
        role_terms=[F.linear(left[r],self.role_weight[:,r]) for r in roles]
        out=torch.stack(role_terms).sum(0)/math.sqrt(len(roles))
        interactions=[F.linear(left[a]*right[b],self.pair_weight[:,i])
             for i,(a,b) in enumerate(self.pairs) if a in views and b in views]
        if interactions:out=out+torch.stack(interactions).sum(0)/math.sqrt(len(interactions))
        return out


class RelationalEventCore(nn.Module):
    def __init__(self,base,width,seed):
        super().__init__();self.base=base;self.composer=ChildRoleComposer(width,seed=seed)
        self.width=width

    def __getattr__(self,name):
        try:return super().__getattr__(name)
        except AttributeError:return getattr(super().__getattr__('base'),name)

    def new_state(self):
        state=self.base.new_state();state.child_views={}
        state.role_projection_madds=0;state.role_pair_products=0;state.role_transport_calls=0
        return state

    def consume_event(self,source,timestamp,content,state):
        mark=torch.as_tensor(content,dtype=self.base.content.weight.dtype,device=self.base.content.weight.device)
        stamp=torch.as_tensor(timestamp,dtype=torch.float64,device=mark.device)
        if mark.shape!=(self.base.content_dim,) or not bool(torch.isfinite(mark).all()):
            raise ValueError('finite native content required before role-cache mutation')
        if not 0<=int(source)<self.base.sources or not bool(torch.isfinite(stamp)) or float(stamp.detach())<state.last_input_time:
            raise ValueError('known source and nondecreasing finite time required')
        if float(mark[self.width+2].detach())>=0:
            # Adjacent-frame children occupy the four relative quadrants.
            # This auxiliary slot keeps the latest view, while the native core
            # still consumes every accepted message. It is not an event log.
            role=2*int(float(mark[self.width].detach())>=.5)+int(float(mark[self.width+1].detach())>=.5)
            group=float(mark[self.width+2].detach())
            state.child_views[(group,role)]=(mark[:self.width],stamp)
            read_time=stamp
            if source in state.contexts:read_time=torch.maximum(read_time,state.contexts[source][1].max())
            views={}
            for (source_group,r),(value,time) in state.child_views.items():
                if source_group!=group:continue
                age=read_time-time
                views[r]=torch.cat([self.base.transport(v,age,self.base.depth-1,h)
                    for h,v in enumerate(value.split(self.base.payload))])
            relation=self.composer(views)
            mark=torch.cat((mark[:self.width]+relation,mark[self.width:]))
            k=len(views);pairs=k*(k-1)//2;rank=self.composer.rank
            state.role_projection_madds+=(3*k-1+pairs)*rank*self.width
            state.role_pair_products+=pairs*rank
            state.role_transport_calls+=k*self.base.heads
        return self.base.consume_event(source,timestamp,mark,state)


class RelationalFrameComposition(FrameNativeComposition):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        # Pixel-native memory already has one sensor address; relation slots
        # are needed only in parents. Keep all32native stages in place.
        for level,obj in enumerate(self.objects[1:],1):
            obj.core=RelationalEventCore(obj.core,self.width,601+level)

    def load_native_state(self,weights):
        mapped={}
        for key,value in weights.items():
            parts=key.split('.')
            if len(parts)>3 and parts[0]=='objects' and int(parts[1])>0 and parts[2]=='core':
                key='.'.join(parts[:3]+['base']+parts[3:])
            mapped[key]=value
        missing=self.load_state_dict(mapped,strict=False)
        if missing.unexpected_keys or any('.core.composer.' not in k for k in missing.missing_keys):
            raise ValueError('native checkpoint migration changed existing parameter coverage')
        return missing

    def work(self,state):
        result=super().work(state)
        private=[p for bank in state.banks[1:] for p in bank.private.values()]
        result.update(role_slots=sum(len(p.child_views) for p in private),
          role_view_tensor_bytes=sum(v.numel()*v.element_size()+t.numel()*t.element_size()
              for p in private for v,t in p.child_views.values()),
          role_projection_madds=sum(p.role_projection_madds for p in private),
          role_pair_products=sum(p.role_pair_products for p in private),
          role_transport_calls=sum(p.role_transport_calls for p in private))
        result['role_boundary']='Role-view bytes are logical retained tensors; aliases/graphs/metadata separate. MAdds cover explicit relation maps; transport/normalization and native learning/optimizer work separate.'
        return result
