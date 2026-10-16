"""Persistent, addressed episodic causes for native language events.

Candidate discovery uses revealed token identity and native race/concept
addresses. Records retain actual successors, never adjacency in a retrieved
list. Capacity and delivered candidates are independent budgets; both paid.
"""
from collections import defaultdict,deque
from itertools import islice
import math
import torch


class NativeCauseBank:
    def __init__(self,depth,width,capacity=32768,suffix=8):
        if min(depth,width,capacity,suffix)<1:raise ValueError('positive bank dimensions required')
        self.depth,self.width,self.capacity,self.suffix=depth,width,capacity,suffix
        self.features=torch.zeros(capacity,depth,width)
        self.tokens=torch.full((capacity,),-1,dtype=torch.long)
        self.successors=torch.full((capacity,),-1,dtype=torch.long)
        self.serials=torch.full((capacity,),-1,dtype=torch.long)
        self.times=torch.zeros(capacity,dtype=torch.float64)
        self.contexts=torch.full((capacity,suffix),-1,dtype=torch.long)
        self.documents=torch.zeros(capacity,dtype=torch.long)
        self.index=defaultdict(deque);self.history=deque(maxlen=suffix)
        self.slot_keys=[() for _ in range(capacity)]
        self.count=0;self.document=0;self.last_time=-math.inf
        self.keys_examined=self.index_entries_examined=self.delivered=self.writes=0

    @torch.no_grad()
    def append(self,token,features,keys,timestamp,*,eos=50256):
        stamp=float(timestamp);token=int(token)
        if token<0 or features.shape!=(self.depth,self.width) or not bool(torch.isfinite(features).all()):
            raise ValueError('revealed token and finite native depth views required')
        if not math.isfinite(stamp) or stamp<self.last_time:raise ValueError('causal event timestamp required')
        previous=self.count-1
        if previous>=0 and int(self.serials[previous%self.capacity])==previous:
            self.successors[previous%self.capacity]=token
        if token==eos:self.document+=1;self.history.clear()
        self.history.append(token);serial=self.count;slot=serial%self.capacity;self.count+=1
        evicted=int(self.serials[slot])
        if evicted>=0:
            for key in self.slot_keys[slot]:
                bucket=self.index.get(key)
                if bucket is None:continue
                while bucket and bucket[0]<=evicted:bucket.popleft();self.index_entries_examined+=1
                if not bucket:del self.index[key]
        self.features[slot].copy_(features.detach().float());self.tokens[slot]=token
        self.successors[slot]=-1;self.serials[slot]=serial;self.times[slot]=stamp;self.documents[slot]=self.document
        self.contexts[slot].fill_(-1);self.contexts[slot,-len(self.history):]=torch.tensor(list(self.history))
        self.slot_keys[slot]=tuple(dict.fromkeys([('token',token),*keys]))
        for key in self.slot_keys[slot]:
            bucket=self.index[key];bucket.append(serial)
            while bucket and bucket[0]<self.count-self.capacity:
                bucket.popleft();self.index_entries_examined+=1
        self.last_time=stamp;self.writes+=1
        return serial

    @torch.no_grad()
    def query(self,token,keys,timestamp,*,maximum=128,recent=8,per_key=2,token_matches=32):
        stamp=float(timestamp)
        if stamp<self.last_time or not math.isfinite(stamp):raise ValueError('query cannot read future bank state')
        if min(maximum,per_key,token_matches)<1 or recent<0:raise ValueError('positive addressed candidate budgets required')
        oldest=max(0,self.count-self.capacity);chosen=[];seen=set()
        def admit(serial):
            self.index_entries_examined+=1
            if serial>=oldest and serial not in seen and int(self.serials[serial%self.capacity])==serial and len(chosen)<maximum:
                seen.add(serial);chosen.append(serial)
        for serial in range(self.count-1,max(oldest-1,self.count-recent-1),-1):admit(serial)
        def read(key,n):
            self.keys_examined+=1;bucket=self.index.get(key)
            if not bucket:return
            while bucket and bucket[0]<oldest:bucket.popleft();self.index_entries_examined+=1
            # Check only a bounded addressed suffix, not all stored events.
            for serial in islice(reversed(bucket),n):admit(serial)
        read(('token',int(token)),token_matches)
        for key in dict.fromkeys(keys):read(key,per_key)
        if not chosen:raise ValueError('no revealed episodic cause available')
        slots=torch.tensor([s%self.capacity for s in chosen]);self.delivered+=len(chosen)
        current=torch.full((self.suffix,),-1,dtype=torch.long)
        current[-len(self.history):]=torch.tensor(list(self.history))
        context=self.contexts[slots];equal=(context==current)&(context>=0)
        matches=equal.flip(-1).long().cumprod(-1).sum(-1)
        return dict(serials=torch.tensor(chosen),features=self.features[slots].clone(),
            tokens=self.tokens[slots].clone(),successors=self.successors[slots].clone(),
            ages=stamp-self.times[slots],suffix_matches=matches,
            same_document=(self.documents[slots]==self.document),valid=torch.ones(len(chosen),dtype=torch.bool))

    def storage(self):
        tensors=[self.features,self.tokens,self.successors,self.serials,self.times,self.contexts,self.documents]
        return dict(capacity=self.capacity,occupied=min(self.count,self.capacity),
                    tensor_bytes=sum(t.numel()*t.element_size() for t in tensors),
                    index_keys=len(self.index),index_references=sum(len(v) for v in self.index.values()),
                    keys_examined=self.keys_examined,index_entries_examined=self.index_entries_examined,
                    delivered_candidates=self.delivered,writes=self.writes,
                    scope='Reserved numeric bank tensors and explicit index counts;Python index/metadata bytes,parameters and credit work additional.')
