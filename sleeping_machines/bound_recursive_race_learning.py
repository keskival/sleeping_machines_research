"""The existing native competing-risk head over an actual bound part pool."""
import torch
from .recursive_race_learning import NativeRecursiveRaceHeads,RaceExposure


class BoundRecursiveRaceHeads(NativeRecursiveRaceHeads):
    def forecast_parts(self,actor,state,level,address,version=None):
        """Pool all declared parts, including bound versions no longer current.

        Identity targets/emission policy are supplied by the learning protocol;
        this method never assigns hidden semantic ids or sees future marks.
        """
        if not 0 <= level < len(self.heads):raise ValueError('unknown forecast level')
        if version is None:version=state.banks[level].published[tuple(address)].version
        key=(level,tuple(address),version)
        parent=state.bindings.entries[key]
        if len(parent.parts)!=self.slots:
            raise ValueError('size this head to the entire declared part pool;no silent truncation')
        if len({p.source[:2] for p in parent.parts})!=self.slots:
            raise ValueError('one binding per source memory required')
        values=[actor.read_bound(state,p.source,state.now) for p in parent.parts]
        context=actor.read_bound(state,key,state.now)[None]
        memory=torch.stack(values)[None,None]
        laws=self.heads[level](memory,context)
        refs=torch.stack([state.bindings.entries[p.source].available for p in parent.parts])[None]
        now=refs.new_tensor([state.now])
        if bool((refs>now[:,None]).any()):raise ValueError('future part in forecast')
        ledger=RaceExposure(laws,refs,now,self.eps)
        return ledger,dict(bound_part_reads=self.slots,parent_context_reads=1,
            scored_race_slots=self.slots,mark_logits=self.slots*self.heads[level].types,
            retained_law_scalars=sum(x.numel() for x in laws.values()),
            boundary='All bound slots scored;native read transport,affine/nonlinear/special functions,backward/optimizer and binding-store lifetimes require separate accounting. No sparse-key or hardware claim.')
