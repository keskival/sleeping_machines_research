"""Native hierarchy with versioned, role-bearing part directories.

A directory is addressable structure beside the evolving native state, not an
inverse of its complete historical computation and not an observation log.
The actor's stochastic transitions, parameters and output remain unchanged.
"""
from dataclasses import dataclass
import torch
from torch import nn
from .relational_frame_composition import RelationalFrameComposition
from .variance_reduced_native_composition import PureTimerObject


@dataclass(frozen=True)
class PartBinding:
    source: tuple  # (level,address,memory version), NOT notification sequence
    issued: torch.Tensor
    pose: tuple
    kind: str


@dataclass(frozen=True)
class BoundSnapshot:
    key: tuple
    available: torch.Tensor
    native: object
    leaf_value: object
    parts: tuple


class BindingStore:
    def __init__(self):
        self.entries = {}
        self.inflight = {}
        self.insertions = self.part_edges = self.reads = self.gc_visits = 0

    def put(self, snapshot):
        if snapshot.key in self.entries: raise ValueError('immutable memory version already exists')
        for part in snapshot.parts:
            if part.source not in self.entries: raise ValueError('unresolved part reference')
            child = self.entries[part.source]
            if part.source[0] >= snapshot.key[0]: raise ValueError('parts must be strictly lower-level memories')
            if float(child.available.detach()) > float(part.issued.detach()) or float(part.issued.detach()) > float(snapshot.available.detach()):
                raise ValueError('part must be available when issued and before parent publication')
        self.entries[snapshot.key] = snapshot
        self.insertions += 1; self.part_edges += len(snapshot.parts)

    def collect(self, roots):
        """Explicit mark/sweep over current roots, pending directories and parts."""
        live = set(); pending = list(roots)
        while pending:
            key = pending.pop()
            if key in live: continue
            if key not in self.entries: raise ValueError('dangling live memory reference')
            live.add(key); self.gc_visits += 1
            pending.extend(p.source for p in self.entries[key].parts)
        old = len(self.entries)
        self.entries = {k:v for k,v in self.entries.items() if k in live}
        return old-len(live)


class BoundTimerObject(PureTimerObject):
    def accept(self, state, address, timestamp, content, event_identity):
        envelope_key = (state.binding_level, address, id(content))
        binding = state.binding_store.inflight.pop(envelope_key)
        pending = super().accept(state,address,timestamp,content,event_identity)
        if pending is not None:
            directory = state.live_parts.setdefault(address,{})
            # A new view refreshes one child binding, rather than counting
            # read/timer/write notifications as three independent children.
            directory[binding.source[:2]] = binding
            state.pending_parts[(address,pending.version)] = tuple(directory[k] for k in sorted(directory))
        return pending

    def publish(self, state, pending, now):
        super().publish(state,pending,now)
        key = (state.binding_level,pending.address,pending.version)
        parts = state.pending_parts.pop((pending.address,pending.version))
        state.binding_store.put(BoundSnapshot(key,pending.ready,state.published[pending.address],None,parts))


class BoundNativeComposition(RelationalFrameComposition):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        # Same native cores and state_dict keys; no parameter or RNG draw.
        self.objects = nn.ModuleList(BoundTimerObject(obj.core,obj.width) for obj in self.objects)

    def new_state(self):
        state = super().new_state(); state.bindings = BindingStore(); state.binding_source = None
        for level,bank in enumerate(state.banks):
            bank.binding_level = level; bank.binding_store = state.bindings
            bank.live_parts = {}; bank.pending_parts = {}
        return state

    def _send(self,state,source_level,source_address,value,version,time,kind,target=None):
        address = tuple(source_address)
        if source_level < 0:
            key = (-1,address,version)
            if key not in state.bindings.entries:
                stamp = torch.as_tensor(time,dtype=torch.float64,device=value.device)
                state.bindings.put(BoundSnapshot(key,stamp,None,value,()))
        else:
            # Timer/query sequence numbers identify computations, while the
            # bound part identifies the actual immutable memory being read.
            key = (source_level,address,state.banks[source_level].published[address].version)
        previous = state.binding_source
        state.binding_source = (key,time,kind)
        try:return super()._send(state,source_level,address,value,version,time,kind,target)
        finally:state.binding_source = previous

    def _enqueue(self,state,time,kind,data):
        if kind == 'deliver':
            if state.binding_source is None: raise ValueError('delivery needs an issued part binding')
            source,issued,event_kind = state.binding_source
            level,address,content,identity = data
            stamp = torch.as_tensor(issued,dtype=torch.float64,device=content.device)
            # Pose is observed frame metadata, not a hidden identity/label.
            pose = tuple(float(x.detach()) for x in content[self.width:self.width+2])
            envelope = (level,address,id(content))
            if envelope in state.bindings.inflight: raise ValueError('duplicate in-flight delivery identity')
            state.bindings.inflight[envelope] = PartBinding(source,stamp,pose,event_kind)
        return super()._enqueue(state,time,kind,data)

    def parts(self,state,level,address,version=None):
        if version is None: version = state.banks[level].published[tuple(address)].version
        key = (level,tuple(address),version)
        state.bindings.reads += 1
        return state.bindings.entries[key].parts

    def read_bound(self,state,key,time):
        """Read the issued native version, transported to an available query time."""
        entry = state.bindings.entries[key]
        stamp = torch.as_tensor(time,dtype=torch.float64,device=entry.available.device)
        if not bool(torch.isfinite(stamp)) or float(stamp.detach()) < float(entry.available.detach()):
            raise ValueError('bound memory is not available at this query time')
        state.bindings.reads += 1
        if key[0] < 0: return entry.leaf_value
        core = self.objects[key[0]].core; record = entry.native
        aligned = core.align(list(record.values.split(core.payload)),record.times,stamp,core.depth-1)
        return core.head(aligned)

    def query_bound(self,state,key,target_level,target_address):
        """Deliver a chosen historical child view through existing native races.

        This is a typed computation/query, not another sensor observation.
        Policy selection is explicit; the method does not claim learned queries.
        """
        if not key[0] < target_level < self.levels:
            raise ValueError('bound queries must travel strictly upwards')
        value = self.read_bound(state,key,state.now)
        state.queries += 1
        previous = state.binding_source
        state.binding_source = (key,state.now,'query')
        try:
            # Bypass latest-version lookup: this query intentionally refers to
            # the version selected through a parent's part directory.
            super()._send(state,key[0],key[1],value,state.queries,state.now,'query',
                          target=(target_level,tuple(target_address)))
        finally:state.binding_source = previous

    def collect_bindings(self,state):
        roots = [(-1,a,v) for a,(_,v,_) in state.inputs.items()]
        for level,bank in enumerate(state.banks):
            roots.extend((level,a,r.version) for a,r in bank.published.items())
            roots.extend(p.source for d in bank.live_parts.values() for p in d.values())
            roots.extend(p.source for d in bank.pending_parts.values() for p in d)
        roots.extend(p.source for p in state.bindings.inflight.values())
        return state.bindings.collect(roots)

    def work(self,state):
        result = super().work(state); store = state.bindings
        # Tensors are aliases of native publications/input values. Count logical
        # retained elements, not additional allocations or autograd graph bytes.
        tensors = [x for e in store.entries.values() for x in
                   ([e.leaf_value] if e.native is None else [e.native.values,e.native.times])]
        result.update(binding_records=len(store.entries),binding_insertions=store.insertions,
                      binding_edges=sum(len(e.parts) for e in store.entries.values()),
                      binding_edge_copies=store.part_edges,binding_reads=store.reads,
                      binding_gc_visits=store.gc_visits,
                      binding_logical_tensor_bytes=sum(x.numel()*x.element_size() for x in tensors))
        result['binding_boundary']='Versioned directory metadata and logical native tensor aliases; Python overhead, extra lifetime, gradients and graph bytes separate. No event-log reconstruction or complete arithmetic claim.'
        return result
