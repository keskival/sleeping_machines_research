"""Executed NumPy arithmetic and logical operands for native trace kernels."""
from collections import defaultdict
from math import prod
import numpy as np


class NumpyLedger:
    def __init__(self):
        self.rows=defaultdict(lambda:dict(calls=0,arithmetic_flops=0,special=0,
            input_bytes=0,output_bytes=0,classification='unknown'))
    def wrap(self,value):
        if isinstance(value,np.ndarray):
            out=value.view(CountedArray);out.ledger=self;return out
        if isinstance(value,(list,tuple)):return type(value)(self.wrap(v) for v in value)
        return value
    def record(self,name,args,out,kind=None):
        arrays=[]
        def collect(v):
            if isinstance(v,np.ndarray):arrays.append(v)
            elif isinstance(v,(tuple,list)):
                for item in v:collect(item)
        collect(args);outputs=[]
        def outputs_of(v):
            if isinstance(v,np.ndarray):outputs.append(v)
            elif isinstance(v,np.generic):outputs.append(np.asarray(v))
            elif isinstance(v,(tuple,list)):
                for item in v:outputs_of(item)
        outputs_of(out);n=sum(v.size for v in outputs);inp=sum(v.size for v in arrays)
        r=self.rows[name];r['calls']+=1;r['input_bytes']+=sum(v.nbytes for v in arrays);r['output_bytes']+=sum(v.nbytes for v in outputs)
        base=name.split('.')[0];f=s=0;classification=kind or 'UNSUPPORTED'
        if base in ['add','subtract','multiply','divide','true_divide','negative','square']:
            f=max(inp-n,0) if name.endswith('.reduce') else n;classification='scalar arithmetic/reduction'
        elif base=='matmul':
            a,b=arrays[:2];k=a.shape[-1];f=2*n*k;classification='matrix contraction,2FLOPs/MAC'
        elif base=='einsum':
            expression=args[0];a,b=arrays[:2]
            reduction={'nij,njp->nip':2,'nij,nj->ni':2,'ni,nip->np':2,'kv,kvp->vp':a.shape[0]}.get(expression)
            if reduction is not None:f=2*n*reduction;classification='trace contraction,2FLOPs/MAC'
        elif base in ['sum','mean']:
            f=max(inp-n,0)+(n if base=='mean' else 0);classification='reduction'
        elif base in ['exp','expm1','log','log1p','sin','cos','sqrt','asinh','logaddexp']:
            s=n;classification='special function'
        elif base in ['absolute','maximum','minimum','amax','amin','max','min','equal','less','greater','isfinite']:
            classification='comparison/control'
        elif base in ['array','asarray','append','stack','concatenate','column_stack','tile','zeros','zeros_like','ones','eye','copy','setitem','reshape']:
            classification='data movement/allocation'
        r['arithmetic_flops']+=int(f);r['special']+=int(s);r['classification']=classification
    def result(self):
        return dict(arithmetic_flops=sum(r['arithmetic_flops'] for r in self.rows.values()),
            special_functions=sum(r['special'] for r in self.rows.values()),
            input_payload_bytes=sum(r['input_bytes'] for r in self.rows.values()),
            output_payload_bytes=sum(r['output_bytes'] for r in self.rows.values()),
            unsupported={k:r for k,r in self.rows.items() if r['classification']=='UNSUPPORTED'},operators=dict(self.rows))


def unwrap(value):
    if isinstance(value,CountedArray):return value.view(np.ndarray)
    if isinstance(value,tuple):return tuple(unwrap(v) for v in value)
    if isinstance(value,list):return [unwrap(v) for v in value]
    if isinstance(value,dict):return {k:unwrap(v) for k,v in value.items()}
    return value


class CountedArray(np.ndarray):
    def __array_finalize__(self,obj):self.ledger=getattr(obj,'ledger',None)
    def __array_ufunc__(self,ufunc,method,*args,**kwargs):
        output=getattr(ufunc,method)(*unwrap(args),**unwrap(kwargs))
        self.ledger.record(ufunc.__name__+('.'+method if method!='__call__' else ''),args,output)
        if kwargs.get('out') is not None:
            return kwargs['out'][0] if len(kwargs['out'])==1 else kwargs['out']
        return self.ledger.wrap(output)
    def __array_function__(self,func,types,args,kwargs):
        output=func(*unwrap(args),**unwrap(kwargs));self.ledger.record(func.__name__,args,output)
        return self.ledger.wrap(output)
    def copy(self,order='C'):
        output=self.view(np.ndarray).copy(order);self.ledger.record('copy',(self,),output)
        return self.ledger.wrap(output)
    def __setitem__(self,index,value):
        super().__setitem__(index,unwrap(value));self.ledger.record('setitem',(value,),self.view(np.ndarray)[index])


class NumpyProxy:
    def __init__(self,ledger):self.ledger=ledger
    def __getattr__(self,name):
        value=getattr(np,name)
        if not callable(value) or isinstance(value,type):return value
        def invoke(*args,**kwargs):
            # Calls with counted arrays dispatch through their public protocol.
            if any(isinstance(x,CountedArray) for x in args):return value(*args,**kwargs)
            output=value(*unwrap(args),**unwrap(kwargs))
            self.ledger.record(name,args,output)
            return self.ledger.wrap(output)
        return invoke
