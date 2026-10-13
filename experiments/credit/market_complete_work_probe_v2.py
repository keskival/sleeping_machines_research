"""B10 ledger with explicit stable-likelihood scalar-work conventions."""
import json,sys
import market_complete_work_probe as probe


class LikelihoodAudit(probe.TorchAudit):
    def formula(self,func,args,kwargs,out):
        name=str(func).split('.')[1].rstrip('_')
        inp=args[0].numel() if args and isinstance(args[0],probe.engine.torch.Tensor) else 0
        if name in ['asinh','special_log_ndtr']:
            return 0,inp,0,'special function; internal approximation unpriced'
        if name=='logsumexp':
            n=out.numel();return 2*inp,inp+n,max(inp-n,0),'stable exp/sum/log conventional formula'
        if name=='logaddexp':
            n=out.numel();return 3*n,2*n,2*n,'stable max/log1p/exp conventional formula'
        if name=='log_sigmoid_forward':
            return 2*inp,2*inp,2*inp,'stable log-sigmoid conventional formula; buffer writes extra'
        if name=='log_sigmoid_backward':
            return 5*inp,0,inp,'cached log-sigmoid derivative conventional formula'
        return super().formula(func,args,kwargs,out)


def main():
    probe.TorchAudit=LikelihoodAudit
    probe.main()
    tag=sys.argv[sys.argv.index('--tag')+1]
    path=probe.engine.ROOT/f'experiments/results/market/{tag}.json'
    record=json.loads(path.read_text())
    record['source_sha256']['experiments/credit/market_complete_work_probe_v2.py']=probe.engine.sha(probe.engine.ROOT/'experiments/credit/market_complete_work_probe_v2.py')
    record['scope']+=' Stable likelihood arithmetic uses declared mathematical conventions; special functions remain separate units, including log-normal CDF implementation. Numerical scalar Python/math work and physical accesses remain outside formula coverage.'
    assert all(not r['torch_work']['unsupported_floating_operators'] and not r['numpy_work']['unsupported'] for r in record['rows'])
    path.write_text(json.dumps(record,indent=2)+'\n')


if __name__=='__main__':main()
