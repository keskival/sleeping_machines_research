"""B10 actual fitting-operation ledger, including NumPy trace work."""
import argparse, copy, json, time, contextlib
import market_native_early_request as engine
from numpy_work_ledger import NumpyLedger,NumpyProxy
from sleeping_machines.operation_audit import OperationAudit
import experiments.market.local_eligibility_state as local
import experiments.credit.local_clock_eligibility as clocks
import experiments.market.projected_clock_state as projected
import experiments.market.paired_clock_credit as paired


class TorchAudit(OperationAudit):
    def formula(self,func,args,kwargs,out):
        name=str(func).split('.')[1].rstrip('_')
        if name=='mv':return 2*args[0].numel(),0,0,'matrix-vector contraction'
        if name=='dot':return 2*args[0].numel(),0,0,'dot contraction'
        if name=='rsub':return out.numel(),0,0,'scalar reverse subtraction'
        if name=='exponential':return 0,0,0,'random sampling, unpriced'
        return super().formula(func,args,kwargs,out)


@contextlib.contextmanager
def instrumentation():
    numpy=NumpyLedger(); proxy=NumpyProxy(numpy); torch=TorchAudit();restores=[]
    for module in [local,clocks,projected,paired]:
        restores.append((module,'np',module.np));module.np=proxy
    original=projected.array;restores.append((projected,'array',original))
    projected.array=lambda tensor:numpy.wrap(original(tensor))
    try:
        with torch:yield torch,numpy
    finally:
        for module,name,value in reversed(restores):setattr(module,name,value)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True)
    ap.add_argument('--mode',choices=['smoke','probe'],required=True);args=ap.parse_args()
    engine.torch.set_num_threads(1);engine.torch.set_default_dtype(engine.torch.float64)
    out=engine.ROOT/f'experiments/results/market/{args.tag}.json';assert not out.exists()
    model,inputs=engine.restored();group,acq,reuse=(8,4,4) if args.mode=='smoke' else (32,64,16)
    engine.torch.manual_seed(19101);critic=engine.NativeMarketCredit(264).double()
    critic.core.head=engine.torch.nn.Linear(critic.core.total_payload,1).double()
    engine.torch.nn.init.zeros_(critic.core.head.weight);engine.torch.nn.init.zeros_(critic.core.head.bias)
    opt=engine.torch.optim.Adam(model.parameters(),lr=.0003);cop=engine.torch.optim.Adam(critic.parameters(),lr=.001)
    t,m,path=engine.load('2026-08-24',128+acq*group);inputs[str(path.relative_to(engine.ROOT))]=engine.sha(path)
    initial=(model,opt,engine.warm(model,t,m,128),critic,cop,.001)
    with instrumentation() as (ta,na):
        bundle,source=engine.stream(initial,t,m,127,acq,group,'online',True)
    source_work=dict(torch=ta.result(),numpy=na.result())
    t,m,path=engine.load('2026-08-25',128+reuse*group);inputs[str(path.relative_to(engine.ROOT))]=engine.sha(path)
    bundle=(*bundle[:2],engine.warm(bundle[0],t,m,128),*bundle[3:]);rows=[]
    for arm in ['own','projected','frozen','online']:
        engine.torch.manual_seed(19111)
        with instrumentation() as (ta,na):
            _,row=engine.stream(bundle,t,m,127,reuse,group,arm)
        engine.torch.manual_seed(19111)
        _,reference=engine.stream(bundle,t,m,127,reuse,group,arm)
        error=max(abs(a-b) for a,b in zip(row['block_nll'],reference['block_nll']))
        assert error<1e-12 and row['requests']==reference['requests']
        if args.mode=='probe':
            parent=engine.ROOT/'experiments/results/market/curie_b10_early_request_20261013_v1_fit.json'
            saved=json.loads(parent.read_text());previous=next(r for r in saved['rows'] if r['arm']==arm)
            assert max(abs(a-b) for a,b in zip(row['block_nll'],previous['block_nll'][:reuse]))<1e-12
            assert row['requests']==previous['requests'][:reuse]
            inputs[str(parent.relative_to(engine.ROOT))]=engine.sha(parent)
        row['torch_work']=ta.result();row['numpy_work']=na.result();row['parity_max_error']=error
        row['covered_arithmetic_flops']=row['torch_work']['arithmetic_flops']+row['numpy_work']['arithmetic_flops']
        row['covered_arithmetic_flops_per_target']=row['covered_arithmetic_flops']/row['targets']
        rows.append(row)
    files=['experiments/credit/market_complete_work_probe.py','experiments/credit/numpy_work_ledger.py',
           'sleeping_machines/operation_audit.py','experiments/credit/market_native_early_request.py',
           'experiments/native_training_work.py','experiments/market/projected_clock_state.py',
           'experiments/market/local_eligibility_state.py','experiments/credit/local_clock_eligibility.py']
    result=dict(status='completed',battle='B10',args=vars(args),acquisition=source,acquisition_work=source_work,rows=rows,
        source_sha256={p:engine.sha(engine.ROOT/p) for p in files},input_sha256=inputs,
        scope='Whole executed bounded acquisition/deployment probe, all controller/proposal/future teacher/actor/gradient/clip/Adam operations inside stream, including numerical trace NumPy kernels. Torch formula coverage and NumPy unsupported operators retained; scalar Python/math and random sampling not fully priced. Logical operand/output exposures are not physical memory traffic. Initialization/checkpoint restoration/day warmup and original prefit outside scope. Instrumented execution parity with uninstrumented replay required; never multiply a512target probe into a measured16384target or fullprefit total. CPU profiling wall has no hardware-performance meaning.')
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    for row in rows:print(json.dumps(dict(arm=row['arm'],nll=row['nll'],covered_flops=row['covered_arithmetic_flops'],
        torch_unknown=list(row['torch_work']['unsupported_floating_operators']),numpy_unknown=list(row['numpy_work']['unsupported']),parity=row['parity_max_error'])),flush=True)


if __name__=='__main__':main()
