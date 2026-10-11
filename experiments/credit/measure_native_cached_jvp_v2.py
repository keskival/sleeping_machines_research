"""B1 numerical and whole primal-plus-probe timing for native cached tangents."""
import argparse,hashlib,json,resource,statistics,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'experiments/tpp'))
from race_tpp_v19 import TemporalMemoryLayer
from native_cached_jvp import NativeCachedJVP

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);args=ap.parse_args()
 torch.set_num_threads(1);torch.set_default_dtype(torch.float64);torch.manual_seed(920);rows=[];start=time.monotonic()
 for d,n,length,queries in [(8,4,2,16),(32,16,8,16),(32,16,32,16)]:
  layer=TemporalMemoryLayer(d,n,0.).eval();u=torch.randn(1,length,d);dt=torch.rand(1,length);dt[:,0]=0
  vectors=torch.randn(queries,*u.shape)
  def forward(z):return layer(z,dt)
  @torch.no_grad()
  def full():return torch.func.vmap(lambda v:torch.func.jvp(forward,(u,),(v,))[1])(vectors).detach()
  def captured():
   cache=NativeCachedJVP.capture(layer,u,dt,context='native',version=0)
   return cache,cache.apply(vectors,context='native',version=0)
  ref=full();cache,value=captured();error=float((ref-value).abs().max());assert error<1e-10
  primal_error=float((cache.output-forward(u)).abs().max());assert primal_error<1e-12
  lam=torch.randn_like(u);xx=u.detach().requires_grad_();yy=forward(xx);reverse=torch.autograd.grad(yy,xx,lam)[0]
  reciprocity=float(((value*lam).flatten(1).sum(1)-(vectors*reverse).flatten(1).sum(1)).abs().max());assert reciprocity<1e-9
  full();captured();full_times=[];capture_times=[];probe_times=[]
  for _ in range(5):
   tick=time.monotonic();full();full_times.append(time.monotonic()-tick)
   tick=time.monotonic();cache=NativeCachedJVP.capture(layer,u,dt,context='native',version=0);capture_times.append(time.monotonic()-tick)
   tick=time.monotonic();cache.apply(vectors,context='native',version=0);probe_times.append(time.monotonic()-tick)
  whole=[a+b for a,b in zip(capture_times,probe_times)]
  with torch.no_grad():next(layer.parameters()).add_(1e-5)
  try:cache.apply(vectors,context='native',version=0)
  except ValueError:pass
  else:raise AssertionError('stale weights accepted')
  rows.append(dict(d=d,modes=n,events=length,directions=queries,input_dimension=u.numel(),cached_floats=cache.cached_floats(),tangent_error=error,primal_error=primal_error,reciprocity_error=reciprocity,full_batched_jvp_s=statistics.median(full_times),capture_s=statistics.median(capture_times),cached_probe_s=statistics.median(probe_times),whole_cached_s=statistics.median(whole),whole_speedup=statistics.median(full_times)/statistics.median(whole),stale_rejection=True))
 sources=['experiments/credit/measure_native_cached_jvp_v2.py','experiments/credit/native_cached_jvp.py','experiments/tpp/race_tpp_v19.py']
 result=dict(status='completed',tag=args.tag,battle='B1/R1 native forward teaching execution',rows=rows,source_sha256={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sources},wall_s=time.monotonic()-start,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scope='Frozen eval native temporal layer, elapsed times fixed, input tangents only,5-repeat warmed median. Reference forward JVP disables unnecessary reverse-graph recording. Capture includes actual primal forward/hooks and extra derivative statistics; whole cache+probe timing charged. Cache/query storage and no parameter/dt/hard-route derivatives explicit. No task quality, learned-transport generalization, TEST or full-learning efficiency claim.')
 with (ROOT/'experiments/results/credit'/f'{args.tag}.json').open('x') as out:out.write(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result),flush=True)
if __name__=='__main__':main()
