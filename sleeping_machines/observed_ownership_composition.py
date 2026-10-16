"""Observed spatial ownership before native part composition.

Static revealed-scene candidate. Threshold must be fitted on TRAIN geometry.
No hidden part identities or labels enter ownership or native messages.
"""
import math
import torch
from .bound_native_composition import BoundNativeComposition


def observed_groups(addresses,threshold):
    points=sorted(set(tuple(p) for p in addresses))
    if not math.isfinite(threshold) or threshold<=0 or not points:raise ValueError('observed pixels and positive finite spatial threshold required')
    parent=list(range(len(points)));work=0
    def root(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for i in range(len(points)):
        for j in range(i):
            work+=1
            if sum((a-b)**2 for a,b in zip(points[i],points[j]))<=threshold**2:
                a,b=root(i),root(j)
                if a!=b:parent[max(a,b)]=min(a,b)
    groups={}
    for i,p in enumerate(points):groups.setdefault(root(i),[]).append(p)
    return list(groups.values()),work


class ObservedOwnershipComposition(BoundNativeComposition):
    def __init__(self,*args,ownership_threshold,**kwargs):
        super().__init__(*args,**kwargs)
        if not math.isfinite(ownership_threshold) or ownership_threshold<=0:raise ValueError('TRAIN-fitted positive spatial threshold required')
        self.ownership_threshold=float(ownership_threshold)

    def new_state(self):
        state=super().new_state();state.ownership=None;state.ownership_distance_tests=0;state.ownership_deliveries=0
        return state

    def establish_ownership(self,state):
        if state.queue:raise ValueError('ownership requires a drained revealed observation scene')
        pixels=state.banks[0].published
        if state.ownership is not None:
            if set(state.ownership)!=set(pixels):raise ValueError('new sensor support requires an explicit ownership revision policy')
            return
        origin=tuple(min(p[i] for p in pixels) for i in range(2))
        groups,work=observed_groups(pixels,self.ownership_threshold)
        ownership={};destinations=set()
        for group in groups:
            low=tuple(min(p[i] for p in group) for i in range(2))
            size=tuple(max(p[i] for p in group)-low[i]+1 for i in range(2))
            # A distinct part memory at its observed spatial centre. Later
            # native levels still use their existing frame/timing/races.
            target=tuple(math.floor((sum(p[i] for p in group)/len(group)-origin[i])/2) for i in range(2))
            if target in destinations:raise ValueError('component-address collision needs a larger addressed namespace')
            destinations.add(target)
            for p in group:ownership[p]=(target,tuple((p[i]-low[i])/size[i] for i in range(2)))
        state.ownership=ownership;state.ownership_distance_tests+=work

    def _enqueue(self,state,time,kind,data):
        if kind=='deliver' and state.binding_source is not None:
            source,_,_=state.binding_source
            if source[0]==0:
                if state.ownership is None:raise ValueError('pixel views need observed ownership before upward emission')
                level,address,content,identity=data
                if level==1:
                    address,pose=state.ownership[source[1]]
                    content=torch.cat((content[:self.width],content.new_tensor(pose),content[self.width+2:]))
                    data=(level,address,content,identity);state.ownership_deliveries+=1
        return super()._enqueue(state,time,kind,data)

    def compose(self,state,address=(0,0),pool_lower=True):
        self.drain(state);self.establish_ownership(state)
        return super().compose(state,address,pool_lower)

    def work(self,state):
        result=super().work(state)
        result.update(ownership_distance_tests=state.ownership_distance_tests,ownership_deliveries=state.ownership_deliveries,
                      observed_part_memories=0 if state.ownership is None else len({a for a,_ in state.ownership.values()}))
        result['ownership_scope']='Static observed spatial groups with a TRAIN-fitted threshold;explicit segmentation/frame policy,not learned semantic identity or general streaming ownership. Pair discovery paid;native messages/races/state preserved.'
        return result
