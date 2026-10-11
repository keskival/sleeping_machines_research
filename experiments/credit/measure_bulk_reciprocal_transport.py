"""B1 native orthogonal block teaching and acquisition/query cost comparison."""
import argparse,hashlib,json,resource,sys,time,statistics
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'experiments/tpp'))
from race_tpp_v19 import TemporalMemoryLayer
from reciprocal_transport import DirectionalPacket
from reciprocal_transport_projection import ProjectedTransport
from bulk_reciprocal_transport import BulkProjectedTransport

def main():
 p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args()
 torch.set_num_threads(1);torch.set_default_dtype(torch.float64);torch.manual_seed(918)
 layer=TemporalMemoryLayer(8,4,0.).eval();x=torch.randn(16);dt=torch.tensor([[0.,.37]]);eye=torch.eye(16);start=time.monotonic()
 def f(z):return layer(z.reshape(1,2,8),dt).flatten()
 def jvp(v):return torch.func.jvp(f,(x,),(v,))[1]
 builders={}
 def sequential():
  t=ProjectedTransport(torch.zeros(16,16),context='cut',version=0)
  for v in eye:t.teach_packet(DirectionalPacket.capture('cut',0,v,jvp(v)))
  return t.matrix
 def bulk():
  responses=torch.func.vmap(jvp)(eye)
  t=BulkProjectedTransport(torch.zeros(16,16),context='cut',version=0)
  t.teach_block(eye,responses,context='cut',version=0);return t.matrix
 def reverse():return torch.func.jacrev(f)(x).T.detach()
 for name,fn in [('sequential_forward',sequential),('bulk_forward',bulk),('cached_exact_reverse',reverse)]:
  fn();times=[]
  for _ in range(5):
   tick=time.monotonic();B=fn();times.append(time.monotonic()-tick)
  tick=time.monotonic();norm=float(torch.linalg.matrix_norm(B,2));bound_s=time.monotonic()-tick
  builders[name]=dict(matrix=B,acquisition_s=statistics.median(times),norm_s=bound_s,norm=norm)
 target=builders['cached_exact_reverse']['matrix'];assert max(float((v['matrix']-target).abs().max()) for v in builders.values())<1e-10
 # Partial block leaves the unobserved orthogonal complement untouched.
 q=eye[:4];initial=torch.randn(16,16);t=BulkProjectedTransport(initial,context='partial',version=0)
 t.teach_block(q,q@target,context='partial',version=0)
 expected=initial+q.T@(q@target-q@initial);assert torch.allclose(t.matrix,expected,atol=1e-12)
 assert torch.allclose((eye-q.T@q)@t.matrix,(eye-q.T@q)@initial,atol=1e-12)
 gamma=.4/max(v['norm'] for v in builders.values());rows=[]
 for count in [1,8,32,128]:
  sources=torch.randn(count,16);reference=None
  for name,entry in builders.items():
   B=entry['matrix'];times=[]
   for _ in range(5):
    tick=time.monotonic();credit=torch.zeros_like(sources)
    for calls in range(1,129):
     residual=sources+gamma*credit@B.T-credit
     if float(torch.linalg.vector_norm(residual,dim=1).max())/.6<=1e-9:break
     credit=credit+residual
    else:raise AssertionError('iteration budget failed')
    times.append(time.monotonic()-tick)
   if reference is None:reference=credit.clone()
   assert float((credit-reference).abs().max())<1e-9
   query_s=statistics.median(times);rows.append(dict(method=name,queries=count,acquisition_s=entry['acquisition_s'],bound_s=entry['norm_s'],queries_s=query_s,total_s=entry['acquisition_s']+entry['norm_s']+query_s,transport_calls=calls,credit_error_vs_first=float((credit-reference).abs().max())))
 sources=['experiments/credit/measure_bulk_reciprocal_transport.py','experiments/credit/bulk_reciprocal_transport.py','experiments/credit/reciprocal_transport.py','experiments/credit/reciprocal_transport_projection.py','experiments/tpp/race_tpp_v19.py']
 result=dict(status='completed',tag=a.tag,battle='B1/R1 transport acquisition economy',rows=rows,gamma=gamma,forward_directions=16,transport_floats=256,source_sha256={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sources},wall_s=time.monotonic()-start,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scope='Two-event frozen native layer, all16 directions per acquisition, batched cotangent queries at identical source/operator/tolerance. Warmed5-repetition median acquisition/query, one measured norm per method. All methods cache a dense16x16 transport. Exact reverse cache is an execution reference, not a new model fit. No task quality, sparse discovery, changing-state reuse, full-model gradient or end-to-end learned-credit scaling claim.')
 with (ROOT/'experiments/results/credit'/f'{a.tag}.json').open('x') as out:out.write(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result),flush=True)
if __name__=='__main__':main()
