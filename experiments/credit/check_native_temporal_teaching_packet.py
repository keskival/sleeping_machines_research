"""B1 native elapsed-time/content transport and reciprocity contracts."""
import argparse,json,time,hashlib,resource,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'experiments/tpp'))
import race_tpp_v19 as native
from native_temporal_teaching_packet import NativeTemporalTeachingPacket

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args();torch.set_num_threads(1);torch.set_default_dtype(torch.float64);torch.manual_seed(941);start=time.monotonic();train=native.load_split('taxi','train');seq=next(x for x in train if len(x[1])>=32);rows=[]
 for length in [4,8,32]:
  layer=native.TemporalMemoryLayer(8,4,0.).eval();u=torch.randn(1,length,8);dt=torch.tensor(seq[0][:length])[None];dt=torch.diff(dt,dim=1,prepend=dt[:,:1]);dt=dt/dt[dt>0].median()
  packet=NativeTemporalTeachingPacket.capture(layer,u,dt,context='taxi',version=0);v=torch.randn(4,*u.shape);w=torch.randn(4,*dt.shape);c=torch.randn(4,*u.shape)
  actual=packet.apply_with_time(v,w,context='taxi',version=0)
  reference=torch.func.vmap(lambda x,tt:torch.func.jvp(lambda z,zz:layer(z,zz),(u,dt),(x,tt))[1])(v,w)
  jvp_error=float((actual-reference).abs().max());x=u.clone().requires_grad_();tt=dt.clone().requires_grad_();y=layer(x,tt)
  exact=torch.autograd.grad(y,(x,tt),c,is_grads_batched=True);content,elapsed=packet.transpose_with_time(c,context='taxi',version=0)
  vjp_error=max(float((content-exact[0]).abs().max()),float((elapsed-exact[1]).abs().max()));reciprocity=float(((actual*c).flatten(1).sum(-1)-(content*v).flatten(1).sum(-1)-(elapsed*w).flatten(1).sum(-1)).abs().max());assert max(jvp_error,vjp_error,reciprocity)<1e-10
  zeros=packet.apply_with_time(v,torch.zeros_like(w),context='taxi',version=0);compatibility=float((zeros-packet.apply(v,context='taxi',version=0)).abs().max());assert compatibility==0
  with torch.no_grad():layer.freq.add_(.001)
  try:packet.apply_with_time(v,w,context='taxi',version=0)
  except ValueError:pass
  else:raise AssertionError('stale temporal packet accepted')
  rows.append(dict(events=length,jvp_error=jvp_error,vjp_error=vjp_error,reciprocity_error=reciprocity,zero_time_compatibility_error=compatibility,cached_floats=packet.cached_floats()))
 sources=['experiments/credit/check_native_temporal_teaching_packet.py','experiments/credit/native_temporal_teaching_packet.py','experiments/credit/native_cached_linearization.py','experiments/credit/native_cached_jvp.py','experiments/tpp/race_tpp_v19.py']
 result=dict(status='completed',tag=a.tag,battle='B1 temporal-credit teaching',rows=rows,source_sha256={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sources},data_sha256=hashlib.sha256((ROOT/'data/easytpp/taxi/train.json').read_bytes()).hexdigest(),wall_s=time.monotonic()-start,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scope='One-layer native recurrence, random content and real Taxi TRAIN elapsed-time patterns, including first zero gap. Joint content/time exact JVP/VJP and reciprocity, stale clock rejection and zero-time compatibility. No hard-route derivative, higher-order/parameter packet, learned-credit utility or efficiency/quality claim.')
 with (ROOT/'experiments/results/credit'/f'{a.tag}.json').open('x') as f:json.dump(result,f,indent=2)
 print(json.dumps(result))
if __name__=='__main__':main()
