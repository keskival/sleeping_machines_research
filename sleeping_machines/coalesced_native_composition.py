"""Persistent native objects integrate locally and emit composed timed views.

A bounded explicit scheduling policy, not a learned emission policy. All raw
receipts update level-zero memory; upper objects consume generated child views.
"""
from .frontier_native_composition import FrontierNativeComposition
from .variance_reduced_native_composition import VarianceReducedNativeComposition


class CoalescedNativeComposition(FrontierNativeComposition):
    def _send(self,state,source_level,source_address,value,version,time,kind,target=None):
        # A local publication makes the indexed memory visible. It is not
        # automatically an upward transmission. Input writes and explicit
        # timer/query deliveries retain the normal native causal path.
        if source_level>=0 and kind=='write':return
        return super()._send(state,source_level,source_address,value,version,time,kind,target)

    def compose(self,state,address=(0,0),pool_lower=True):
        self.drain(state)
        if pool_lower:
            for level in range(self.levels-1):
                children=sorted(state.banks[level].published)
                for child in children:
                    root=tuple(v//2**(self.levels-1-level) for v in child)
                    if root==tuple(address):self.arm(state,level,child)
                # All children arm at the same current time. Their native
                # clocks determine arrival order at parents; no sorted-time
                # replacement or argmax route is introduced.
                self.drain(state)
        return VarianceReducedNativeComposition.compose(self,state,address,pool_lower=False)
