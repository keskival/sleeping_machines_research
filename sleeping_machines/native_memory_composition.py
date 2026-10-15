"""Bounded multi-memory composition using the native temporal object primitive.

No new encoder or attention network: higher memory learns from typed responses
from indexed lower memories, then generates its answer at a native race time.
"""
import heapq
import math
import torch
from .native_object_hierarchy import NativeObjectHierarchy


class NativeMemoryComposition(NativeObjectHierarchy):
    def _enqueue(self,state,time,kind,data):
        # Python times must not round through float32 before a causal comparison.
        stamp=float(torch.as_tensor(time,dtype=torch.float64).detach())
        if not math.isfinite(stamp) or stamp<state.now:
            raise ValueError('no scheduling into the observed past')
        state.serial+=1;heapq.heappush(state.queue,(stamp,state.serial,time,kind,data))

    def drain(self,state):
        # Every publication travels strictly upward; no automatic timer rearming.
        # A finite external/query set therefore gives a finite event queue.
        while state.queue:
            self.advance(state,state.queue[0][0])
        return state

    def compose(self,state,address=(0,0),pool_lower=True):
        """One explicit read round; learned sparse query policy remains future work.

Only learned lower memories are queried, not a direct raw-input shortcut. All
active lower addresses in this root's geometric support participate. Dormant
objects are not swept. Query replies are computation, not new observations.
"""
        self.drain(state)
        if pool_lower:
            for level,bank in enumerate(state.banks[:-1]):
                for child in sorted(bank.published):
                    ancestor=tuple(v//(2**(self.levels-1-level)) for v in child)
                    if ancestor==tuple(address):
                        self.query(state,level,child,self.levels-1,address)
            self.drain(state)
        obj=self.objects[-1];bank=state.banks[-1]
        timer=obj.arm(bank,tuple(address),state.now)
        fired=obj.fire(bank,timer,timer.due)
        value,_=fired;state.timer_emissions+=1
        state.now=float(timer.due.detach())
        return self.head(value),timer.due
