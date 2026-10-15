"""Compose stabilized adjacent learned memories before asking their parent.

Explicit capability construction, not a measured efficiency improvement. It
retains event-native races and indexed state; queries are geometric, not learned.
"""
from .variance_reduced_native_composition import VarianceReducedNativeComposition


class FrontierNativeComposition(VarianceReducedNativeComposition):
    def compose(self,state,address=(0,0),pool_lower=True):
        self.drain(state)
        if pool_lower:
            # Freeze each frontier's address list before its read round. All
            # children publish their composed state before the next level reads.
            for level in range(1,self.levels):
                lower=state.banks[level-1]
                for child in sorted(lower.published):
                    root=tuple(v//2**(self.levels-level) for v in child)
                    if root==tuple(address):
                        parent=tuple(v//2 for v in child)
                        self.query(state,level-1,child,level,parent)
                self.drain(state)
        # Reuse native race-generated root readout, no additional global reads.
        return super().compose(state,address,pool_lower=False)
