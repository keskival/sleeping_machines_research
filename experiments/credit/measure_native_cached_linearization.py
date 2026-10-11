"""B1 exact shared primal-packet tangent/cotangent execution contract."""
import argparse,hashlib,json,resource,statistics,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'experiments/tpp'))
from race_tpp_v19 import TemporalMemoryLayer
from native_cached_linearization import NativeCachedLinearization

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);args=ap.parse_args()
 torch.set_num_threads(1);torch.set_default_dtype(torch.float64);torch.manual_seed(921);start=time.monotonic();rows=[]
 for d,n,length in [(8,4,2),(32,16,8),(32,16,32)]:
  layer=TemporalMemoryLayer(d,n,0.).eval();u=torch.randn(1,length,d);dt=torch.rand(1,length);dt[:,0]=0;q=16
  directions=torch.randn(q,*u.shape);cotangents=torch.randn_like(directions)
  def forward(x):return layer(x,dt)
  def automatic():
   x=u.detach().requires_grad_();y=forward(x)
   return torch.autograd.grad(y,x,cotangents,is_grads_batched=True)[0].detach()
  def cached():
   cache=NativeCachedLinearization.capture(layer,u,dt,context='native',version=0)
   return cache,cache.transpose(cotangents,context='native',version=0)
  ref=automatic();cache,answer=cached();error=float((ref-answer).abs().max());assert error<1e-10
  tangent=cache.apply(directions,context='native',version=0)
  reciprocity=float(((tangent*cotangents).flatten(1).sum(1)-(directions*answer).flatten(1).sum(1)).abs().max());assert reciprocity<1e-9
  automatic();cached();auto_times=[];capture_times=[];reverse_times=[];both_times=[]
  for _ in range(5):
   tick=time.monotonic();automatic();auto_times.append(time.monotonic()-tick)
   tick=time.monotonic();cache=NativeCachedLinearization.capture(layer,u,dt,context='native',version=0);capture_times.append(time.monotonic()-tick)
   tick=time.monotonic();cache.transpose(cotangents,context='native',version=0);reverse_times.append(time.monotonic()-tick)
   tick=time.monotonic();cache.apply(directions,context='native',version=0);cache.transpose(cotangents,context='native',version=0);both_times.append(time.monotonic()-tick)
  whole=statistics.median([a+b for a,b in zip(capture_times,reverse_times)])
  with torch.no_grad():next(layer.parameters()).add_(1e-5)
  try:cache.transpose(cotangents,context='native',version=0)
  except ValueError:pass
  else:raise AssertionError('stale reverse packet accepted')
  rows.append(dict(d=d,modes=n,events=length,cotangents=q,input_dimension=u.numel(),cached_floats=cache.cached_floats(),dense_jacobian_floats=u.numel()**2,cotangent_error=error,reciprocity_error=reciprocity,auto_forward_plus_vjp_s=statistics.median(auto_times),capture_s=statistics.median(capture_times),cached_vjp_s=statistics.median(reverse_times),cached_forward_plus_vjp_s=whole,whole_reverse_speedup=statistics.median(auto_times)/whole,shared_capture_plus_both_s=statistics.median([a+b for a,b in zip(capture_times,both_times)])))
 sources=['experiments/credit/measure_native_cached_linearization.py','experiments/credit/native_cached_linearization.py','experiments/credit/native_cached_jvp.py','experiments/tpp/race_tpp_v19.py']
 result=dict(status='completed',tag=args.tag,battle='B1/R1 native shared teaching/credit packet',rows=rows,source_sha256={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sources},wall_s=time.monotonic()-start,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scope='Frozen eval native layer, fixed elapsed times, input JVP/VJP only. Both primal-plus-reverse arms batch16 cotangents; warmed5-repeat medians. No dense cut Jacobian in cached implementation; reported cache floats exclude shared model, input/dt and query/output buffers. Autograd memory not measured. No parameter/dt/hard-route credit, learned-transport generalization, TEST or full-fitting/quality win.')
 with (ROOT/'experiments/results/credit'/f'{args.tag}.json').open('x') as out:out.write(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result),flush=True)
if __name__=='__main__':main()
