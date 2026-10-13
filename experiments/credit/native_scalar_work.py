"""Declared scalar algorithms alongside registered matrix FLOPs.

Nonlinear functions and comparisons are separate units, never one-FLOP proxies.
Unknown operators stay uncovered. Counts describe algorithms, not silicon energy.
"""
from collections import Counter
import torch


def tensor_values(value):
    if isinstance(value,torch.Tensor):yield value
    elif isinstance(value,(list,tuple)):
        for x in value:yield from tensor_values(x)
    elif isinstance(value,dict):
        for x in value.values():yield from tensor_values(x)


def scalar_work(func,args,kwargs,out,registered):
    name=str(func).split('.')[1];base=name.removeprefix('_foreach_').rstrip('_')
    outputs=[t for t in tensor_values(out) if t.is_floating_point()]
    n=sum(t.numel() for t in outputs)
    inputs=[t for t in tensor_values(args) if t.is_floating_point()]
    work=Counter();kind='uncovered'
    if func.overloadpacket in registered:
        return 'registered_matrix_or_existing_rule',work
    if base in ('add','sub','mul','div','neg','reciprocal'):
        work['scalar_arithmetic']=n*(2 if base in ('add','sub') and kwargs.get('alpha',1)!=1 else 1);kind='scalar_rule'
    elif base in ('addcmul','addcdiv'):
        work['scalar_arithmetic']=n*(2 if kwargs.get('value',1)==1 else 3);kind='scalar_rule'
    elif base=='lerp':work['scalar_arithmetic']=3*n;kind='scalar_rule'
    elif base in ('square',):work['scalar_arithmetic']=n;kind='scalar_rule'
    elif base=='pow' and len(args)>1 and isinstance(args[1],(int,float)) and args[1]==2:
        work['scalar_arithmetic']=n;kind='scalar_rule'
    elif base in ('exp','expm1','log','log1p','sqrt','rsqrt','sin','cos','tanh'):
        work['nonlinear_'+base]=n;kind='nonlinear_rule'
    elif base=='sigmoid':
        work['scalar_arithmetic']=3*n;work['nonlinear_exp']=n;kind='declared_sigmoid_algorithm'
    elif base=='softplus':
        work['scalar_arithmetic']=n;work['nonlinear_exp']=n;work['nonlinear_log1p']=n;kind='declared_softplus_algorithm'
    elif base in ('eq','ne','lt','le','gt','ge','clamp','clamp_min','clamp_max','maximum','minimum'):
        total=sum(t.numel() for t in tensor_values(out));work['comparisons']=total*(2 if base=='clamp' else 1);kind='comparison_rule'
    elif base in ('sum','mean') and inputs:
        total=inputs[0].numel();work['scalar_arithmetic']=max(0,total-n)+(n if base=='mean' else 0);kind='reduction_rule'
    elif base in ('linalg_vector_norm','norm') and inputs:
        # L2 only; other p-norms must carry their own declared rule.
        order=kwargs.get('ord',args[1] if len(args)>1 else 2)
        if order==2:
            total=sum(t.numel() for t in inputs);work['scalar_arithmetic']=max(0,2*total-n);work['nonlinear_sqrt']=n;kind='declared_l2_algorithm'
    elif base in ('_softmax','_log_softmax') and inputs:
        x=inputs[0];dim=args[1];rows=x.numel()//x.shape[dim]
        work['scalar_arithmetic']=3*x.numel()-rows;work['comparisons']=x.numel()-rows
        work['nonlinear_exp']=x.numel()
        if base=='_log_softmax':work['nonlinear_log']=rows
        kind='declared_stable_normalization_algorithm'
    elif base in ('_softmax_backward_data','_log_softmax_backward_data') and inputs:
        x=inputs[0];dim=args[2];rows=x.numel()//x.shape[dim]
        work['scalar_arithmetic']=(4 if base=='_softmax_backward_data' else 3)*x.numel()-rows
        if base=='_log_softmax_backward_data':work['nonlinear_exp']=x.numel()
        kind='declared_normalization_backward_algorithm'
    elif base in ('view','_unsafe_view','as_strided','transpose','t','permute','detach','alias','slice','select','unsqueeze','squeeze','expand','empty','empty_like','empty_strided','zeros','zeros_like','ones','ones_like','copy','clone','copy','fill','zero','index','index_select','index_put','index_put','gather','scatter','where','cat','stack','split','unbind','to','_to_copy','_local_scalar_dense'):
        kind='movement_or_metadata_no_arithmetic_rule'
    return kind,work


def validate_scalar_rules():
    x=torch.zeros(4);y=torch.ones(4)
    tests=[(torch.ops.aten.add.Tensor,(x,y),{},dict(scalar_arithmetic=4)),
           (torch.ops.aten.addcmul.default,(x,y,y),{},dict(scalar_arithmetic=8)),
           (torch.ops.aten.lerp.Scalar,(x,y,.5),{},dict(scalar_arithmetic=12)),
           (torch.ops.aten._foreach_mul.Scalar,([x,y],.5),{},dict(scalar_arithmetic=8)),
           (torch.ops.aten._foreach_norm.Scalar,([x,y],2),{},dict(scalar_arithmetic=14,nonlinear_sqrt=2)),
           (torch.ops.aten.sum.default,(y,),{},dict(scalar_arithmetic=3))]
    for op,args,kwargs,expected in tests:
        out=op(*args,**kwargs);kind,actual=scalar_work(op,args,kwargs,out,{})
        assert dict(actual)==expected,(str(op),actual,expected)
    return dict(cases=len(tests),gate='PASS',scope='Declared scalar algorithm units, not fused instruction counts')
