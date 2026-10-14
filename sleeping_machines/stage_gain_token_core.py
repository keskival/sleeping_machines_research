"""R1 per-stage residual gains without changing frozen packed kernels.

A prerequisite for controlled depth growth, not an identity-growth model.
Gains are fixed configuration, checkpointed independently of tensor dtype.
"""
import copy
import math
from .packed_token_core import PackedTokenCore


class StageGainTokenCore(PackedTokenCore):
    def __init__(self,original):
        gains=[]
        for layer in original.units:
            values=[float(u.gain) for head in layer for pool in head for u in pool]
            if len(set(values))!=1:
                raise ValueError('Each stage requires one gain across its receivers and heads')
            gains.append(values[0])
        self._validate(gains,original.depth)
        # The old packer requires a uniform gain. Normalize only an isolated
        # construction copy, then restore the actual gains in our bank views.
        normalized=copy.deepcopy(original)
        for layer in normalized.units:
            for head in layer:
                for pool in head:
                    for unit in pool:unit.gain=gains[0]
        super().__init__(normalized)
        self.stage_gains=tuple(gains)

    @classmethod
    def from_packed(cls,source):
        """Clone a completed packed core without reconstructing its unit graph.

        Parameters/state are storage independent; optimizer state is external
        and must be transferred explicitly by a future continuation owner.
        """
        if type(source) is not PackedTokenCore:
            raise ValueError('Explicit migration requires an unmodified legacy PackedTokenCore')
        gains=(float(source.gain),)*source.depth
        cls._validate(gains,source.depth)
        result=copy.deepcopy(source)
        result.__class__=cls
        result.stage_gains=gains
        return result

    @staticmethod
    def _validate(gains,depth):
        if len(gains)!=depth or any(not math.isfinite(g) or g<0 for g in gains):
            raise ValueError('One finite nonnegative gain per temporal stage required')

    def _stacked(self,source):
        layers=super()._stacked(source)
        for layer,gain in zip(layers,self.stage_gains):layer['gain']=gain
        return layers

    def get_extra_state(self):
        return dict(schema=1,stage_gains=self.stage_gains)

    def set_extra_state(self,state):
        if state.get('schema')!=1:raise ValueError('Unknown stage-gain checkpoint schema')
        gains=tuple(float(g) for g in state['stage_gains'])
        self._validate(gains,self.depth)
        self.stage_gains=gains
