"""Executed aten ledger: arithmetic, special functions and comparisons separately.

Fused kernels use explicit scalar-algorithm conventions, not a backward
multiplier. Unknown operators fail closed. Metadata/memory movement is recorded
as calls and output elements but is not labelled floating point arithmetic.
"""
from collections import Counter
import math
import torch
from torch.utils._python_dispatch import TorchDispatchMode

def tensors(x):
    if isinstance(x,torch.Tensor):yield x
    elif isinstance(x,(list,tuple)):
        for y in x:yield from tensors(y)

def size(x):return sum(t.numel() for t in tensors(x))
class Ledger(TorchDispatchMode):
    def __init__(self):
        super().__init__();self.calls=Counter();self.flops=0;self.special=0;self.comparisons=0;self.unsupported=Counter()
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        kw=kwargs or {}; name=str(func).split('.')[1];out=func(*args,**kw);n=size(out)
        self.calls[str(func)]+=1
        metadata={'view','reshape','_unsafe_view','transpose','t','permute','slice','select','unsqueeze','squeeze','expand','detach','alias','clone','copy_','_to_copy','to','empty','empty_like','empty_strided','zeros','zeros_like','ones','ones_like','full','full_like','new_zeros','new_empty','new_empty_strided','cat','stack','split','split_with_sizes','unbind','as_strided','as_strided_','resize_','lift_fresh','index','index_put','index_put_','index_select','gather','scatter','scatter_','slice_scatter','select_scatter','fill_','zero_','arange','scalar_tensor','_local_scalar_dense','embedding','triu','tril','is_same_size','resolve_conj','resolve_neg','contiguous','repeat'}
        unary={'neg','abs','sign','ceil','floor','reciprocal'}
        binary={'add','add_','sub','sub_','mul','mul_','div','div_','true_divide','lerp','lerp_','remainder'}
        specials={'exp','exp_','expm1','log','log_','log1p','sqrt','sqrt_','rsqrt','sin','cos','tanh','sigmoid','special_log_ndtr','erf','pow','pow_'}
        comparisons={'gt','ge','lt','le','eq','ne','maximum','minimum','clamp','clamp_min','clamp_max','clamp_','clamp_min_','clamp_max_','isfinite','isinf','isnan','logical_and','logical_or','bitwise_and','bitwise_or','any','all','max','min','amin','amax','argmax','aminmax'}
        if str(func).startswith('profiler.'):pass
        elif name in metadata:pass
        elif name in unary:self.flops+=n
        elif name in binary:self.flops+=n*(3 if name.startswith('lerp') else 1)
        elif name in specials:self.special+=n
        elif name in comparisons:self.comparisons+=size(args[0])
        elif name in ('where','masked_fill','masked_fill_'):self.comparisons+=n
        elif name in ('mm','bmm','addmm'):
            a,b=(args[1],args[2]) if name=='addmm' else args[:2]
            k=a.shape[-1];self.flops+=n*(2*k-1)+(n if name=='addmm' else 0)
        elif name in ('sum','mean'):
            self.flops+=max(0,size(args[0])-n)+(n if name=='mean' else 0)
        elif name=='addcmul_' or name=='addcdiv_':self.flops+=3*n
        elif name in ('_softmax','_log_softmax','logsumexp'):
            a=args[0];dim=args[1]
            if isinstance(dim,(list,tuple)):dim=dim[0]
            count=a.numel(); groups=n if name=='logsumexp' else count//a.shape[dim]
            self.flops+=3*count-groups;self.special+=count+(groups if name!='_softmax' else 0)
            self.comparisons+=count-groups
        elif name in ('_softmax_backward_data','_log_softmax_backward_data'):
            self.flops+=4*n if name=='_softmax_backward_data' else 2*n;self.special+=n if name=='_log_softmax_backward_data' else 0
        elif name=='native_layer_norm':
            a=args[0];width=math.prod(args[1]);count=a.numel();groups=count//width
            self.flops+=8*count+2*groups;self.special+=groups
        elif name=='native_layer_norm_backward':
            count=args[0].numel();width=math.prod(args[2]);groups=count//width
            self.flops+=12*count+3*groups
        elif name in ('gelu','gelu_backward'):
            self.flops+=size(args[0])*(5 if name=='gelu' else 12);self.special+=size(args[0])*(1 if name=='gelu' else 2)
        elif name in ('sigmoid_backward','tanh_backward'):self.flops+=3*n
        elif name in ('softplus','softplus_backward','log_sigmoid_forward','log_sigmoid_backward'):
            count=args[0].numel();self.flops+=3*count;self.special+=2*count;self.comparisons+=count
        elif name=='logaddexp':self.flops+=3*n;self.special+=2*n;self.comparisons+=n
        elif name=='embedding_dense_backward':self.flops+=args[0].numel()
        elif name=='scatter_add':self.flops+=args[3].numel()
        elif name in ('slice_backward','select_backward'):pass
        elif name=='bernoulli_':self.comparisons+=n # random generation recorded, not FLOPs
        elif name=='linalg_vector_norm':
            count=args[0].numel();self.flops+=2*count-n;self.special+=n
        else:raise RuntimeError('Unaccounted operator: '+str(func))
        return out
    def record(self):return dict(arithmetic=self.flops,special_functions=self.special,comparisons=self.comparisons,operator_calls=dict(self.calls),unsupported=dict(self.unsupported),validated=not bool(self.unsupported),convention='aten_scalar_algorithms_v3')
