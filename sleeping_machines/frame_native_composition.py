"""World-indexed native pixel memories, object-relative upper composition.

Anchor only after the revealed observation queue drains. No sensor replay,
future observation, part identity or new parameter is introduced. Frame policy
is explicit and scene-specific, not learned general-purpose object discovery.
"""
import torch
from .coalesced_native_composition import CoalescedNativeComposition
from .variance_reduced_native_composition import VarianceReducedNativeComposition


class FrameNativeComposition(CoalescedNativeComposition):
    def new_state(self):
        state=super().new_state();state.frame_origin=None
        state.frame_bound_comparisons=0;state.frame_coordinate_subtractions=0
        return state

    def _send(self,state,source_level,source_address,value,version,time,kind,target=None):
        if source_level>=0 and kind=='write':return
        if source_level==-1 and kind=='write' and target is None:
            # Preserve each world pixel's native temporal memory independently.
            level,address,pose=0,tuple(source_address),[0.,0.]
        else:
            if state.frame_origin is None:raise ValueError('compose the revealed scene to establish its frame')
            level=source_level+1 if target is None else target[0]
            if level>=self.levels:return
            local=tuple(source_address)
            if source_level<=0:
                local=tuple(v-o for v,o in zip(local,state.frame_origin))
                state.frame_coordinate_subtractions+=2
            address=tuple(v//2 for v in local) if target is None else tuple(target[1])
            source_scale=2**max(0,source_level);target_scale=2**level
            pose=[(local[i]*source_scale-address[i]*target_scale)/target_scale for i in range(2)]
        types={'write':[1.,0.,0.],'query':[0.,1.,0.],'timer':[0.,0.,1.]}
        metadata=value.new_tensor([*pose,source_level/max(1,self.levels),*types[kind]])
        identity=((source_level,tuple(source_address),kind),version)
        self._enqueue(state,time,'deliver',(level,address,torch.cat((value,metadata)),identity))

    def compose(self,state,address=(0,0),pool_lower=True):
        self.drain(state)
        if state.frame_origin is None:
            pixels=state.banks[0].published
            if not pixels:raise ValueError('at least one revealed pixel required')
            state.frame_origin=tuple(min(p[i] for p in pixels) for i in range(2))
            state.frame_bound_comparisons+=2*(len(pixels)-1)
        if pool_lower:
            for level in range(self.levels-1):
                for child in sorted(state.banks[level].published):
                    local=(tuple(v-o for v,o in zip(child,state.frame_origin)) if level==0 else child)
                    if level==0:state.frame_coordinate_subtractions+=2
                    root=tuple(v//2**(self.levels-1-level) for v in local)
                    if root==tuple(address):self.arm(state,level,child)
                self.drain(state)
        return VarianceReducedNativeComposition.compose(self,state,address,pool_lower=False)

    def work(self,state):
        result=super().work(state)
        result.update(pixel_native_memories=len(state.banks[0].published),
          frame_bound_comparisons=state.frame_bound_comparisons,
          frame_coordinate_subtractions=state.frame_coordinate_subtractions)
        return result
