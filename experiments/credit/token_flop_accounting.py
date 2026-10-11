"""Charge dense vocabulary matrix-vector products omitted by Torch's registry."""
import torch
from torch.utils.flop_counter import FlopCounterMode

def mv_flops(matrix_shape,vector_shape,*args,out_shape=None,**kwargs):
    if len(matrix_shape)!=2 or len(vector_shape)!=1 or matrix_shape[1]!=vector_shape[0]:
        raise ValueError('Invalid matrix-vector shapes')
    return 2*matrix_shape[0]*matrix_shape[1]

class TokenFlopCounter(FlopCounterMode):
    def __init__(self,display=False):
        super().__init__(display=display,custom_mapping={torch.ops.aten.mv:mv_flops})
