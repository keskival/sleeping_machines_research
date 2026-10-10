"""Bounded-memory operation accounting for the v2 native integration gate.

Unlike the event profiler, streaming counters retain no per-operation trace.
Count supported arithmetic separately from the complete operator invocation
ledger; unsupported/special-function work is not silently called zero FLOPs.
"""
import argparse
import hashlib
import json
import resource
import socket
import time

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.flop_counter import FlopCounterMode
from torch.utils._pytree import tree_flatten
from reciprocal_native_v2 import contracts, smoke
from reciprocal_native_v1 import ROOT, DT


class OperatorLedger(TorchDispatchMode):
    def __init__(self):
        super().__init__();self.rows={}

    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        result=func(*args,**(kwargs or {}))
        name=str(func)
        row=self.rows.setdefault(name,dict(calls=0,output_elements=0))
        row['calls']+=1
        leaves,_=tree_flatten(result)
        row['output_elements']+=sum(x.numel() for x in leaves if isinstance(x,torch.Tensor))
        return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True)
    parser.add_argument('--mode',choices=('contract','smoke'),required=True)
    parser.add_argument('--depth',type=int,default=2);parser.add_argument('--steps',type=int,default=3)
    args=parser.parse_args();out=ROOT/'experiments/results/credit'/(args.tag+'.json')
    if out.exists():raise FileExistsError(out)
    torch.set_num_threads(1);torch.set_default_dtype(DT);torch.manual_seed(165)
    start=time.monotonic();ledger=OperatorLedger()
    with FlopCounterMode(display=False) as counter,ledger:
        result=contracts() if args.mode=='contract' else smoke(args.depth,args.steps)
    paths=['experiments/credit/reciprocal_native_v3.py','experiments/credit/reciprocal_native_v2.py',
           'experiments/credit/reciprocal_native_v1.py','experiments/tpp/recall_tpp_v4.py','experiments/tpp/race_tpp_v5.py']
    record=dict(status='completed',battle='R1/B1',tag=args.tag,args=vars(args),metrics=result,
                decision='Whether selective teacher/credit/update gates pass under bounded memory across depth',
                training=args.mode=='smoke',wall_s=time.monotonic()-start,host=socket.gethostname(),
                peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                supported_flops=counter.get_total_flops(),operator_ledger=ledger.rows,
                counting_scope='Streaming PyTorch supported arithmetic plus all dispatched operator calls/output elements; special and unsupported arithmetic require separate cost formulas. Wall includes instrumentation. No total-FLOP efficiency claim.',
                source_sha256={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in paths})
    out.parent.mkdir(parents=True,exist_ok=True);tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(record,indent=2)+'\n');tmp.replace(out)
    print('RESULT',json.dumps({k:v for k,v in record.items() if k!='operator_ledger'}),flush=True)


if __name__=='__main__':main()
