"""B1 complete native actor-step cost, including packet plus retained graph."""
import argparse,hashlib,json,resource,statistics,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'experiments/tpp'))
import race_tpp_v19 as native
from connected_deep_teaching_actor import ConnectedDeepTeachingActor as PacketActor

class ProbeReference(native.RaceTPP):
 def encode(self,t,m,mask):
  dt=torch.diff(t,dim=1,prepend=t[:,:1]).clamp_min(0)/self.scale;age=(t-t[:,:1])/self.scale
  x=self.embed(m)+self.gap(torch.stack([torch.log1p(dt),(dt>0).to(t.dtype),torch.log1p(age)],-1));self.probe_input=x.detach();self.probe_dt=dt
  h=x
  for layer in self.layers:h=layer(h,dt)
  return h,self.marks(h,m,dt)

def deep_forward(model,x,dt):
 for layer in model.layers:x=layer(x,dt)
 return x

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args()
 gate_path=ROOT/'experiments/results/credit/aws_native_packet_actor_update_20261011_v1.json';gate=json.loads(gate_path.read_text());assert gate['status']=='completed'
 for s,h in gate['source_sha256'].items():assert hashlib.sha256((ROOT/s).read_bytes()).hexdigest()==h
 torch.set_num_threads(1);torch.set_default_dtype(torch.float64);torch.manual_seed(923);start=time.monotonic()
 train=native.load_split('taxi','train');K=1+max(int(m.max()) for _,m in train);rows=[]
 for batch,length,d,modes,queries,depth in [(4,32,64,32,q,depth) for depth in [8,32] for q in [1,16]]:
  selected=[(t[:length],m[:length]) for t,m in train if len(m)>=length][:batch];assert len(selected)==batch
  t,m,mask=next(native.batches(selected,batch,False,__import__('random').Random(0)))
  gaps=np.concatenate([np.diff(ti) for ti,mi in selected]);pos=gaps[gaps>0];scale=float(np.median(pos));qs=np.log(np.quantile(pos,[.1,.5,.9])).tolist();args=(K,d,modes,depth,2,3,2,0.,scale,qs)
  ref=ProbeReference(*args).eval();actor=PacketActor(*args).eval();actor.load_state_dict(ref.state_dict());ropt=torch.optim.Adam(ref.parameters(),lr=.001);opt=torch.optim.Adam(actor.parameters(),lr=.001)
  rtimes=[];ptimes=[];errors=[];probe_errors=[];directions=torch.randn(queries,*t.shape,d)
  for version in range(8):
   # Alternate timing order, identical data/updates; allocations and Adam included.
   for name in (['reference','packet'] if version%2==0 else ['packet','reference']):
    model,optimizer=(ref,ropt) if name=='reference' else (actor,opt);actor.credit_version=version
    tick=time.monotonic();optimizer.zero_grad();tl,ml,count,_=model.loglik(t,m,mask);loss=-(tl+ml)/count;loss.backward()
    if name=='packet':responses=actor.packet.apply(directions,context='taxi',version=version)
    else:
     with torch.no_grad():responses=torch.func.vmap(lambda v:torch.func.jvp(lambda x:deep_forward(ref,x,ref.probe_dt),(ref.probe_input,),(v,))[1])(directions)
    if name=='reference':reference_responses=responses.detach().clone()
    else:packet_responses=responses.detach().clone()
    if name=='reference':reference_gradients=[p.grad.detach().clone() for p in model.parameters()]
    else:packet_gradients=[p.grad.detach().clone() for p in model.parameters()]
    optimizer.step();elapsed=time.monotonic()-tick
    if version>=3:(rtimes if name=='reference' else ptimes).append(elapsed)
   gradient_error=max(float((g-h).abs().max()) for g,h in zip(reference_gradients,packet_gradients));assert gradient_error<1e-10
   adam_error=max(float((ropt.state[p][k]-opt.state[q][k]).abs().max()) for p,q in zip(ref.parameters(),actor.parameters()) for k in ['exp_avg','exp_avg_sq']);assert adam_error<1e-10
   probe_error=float((reference_responses-packet_responses).abs().max());assert probe_error<1e-10;probe_errors.append(probe_error)
   error=max(float((p-q).detach().abs().max()) for p,q in zip(ref.parameters(),actor.parameters()));assert error<1e-8;errors.append(error)
  r=statistics.median(rtimes);p=statistics.median(ptimes);rows.append(dict(depth=depth,teaching_directions=queries,final_gradient_error=gradient_error,final_adam_error=adam_error,max_probe_error=max(probe_errors),batch=batch,events=length,d=d,modes=modes,targets=int(mask[:,1:].sum()),reference_step_s=r,packet_step_s=p,whole_step_speedup=r/p,max_parameter_error=max(errors),parameters=sum(p.numel() for p in ref.parameters()),scope='5 timed steps after3 warmups, same data/order/Adam; alternating timing order'))
 sources=['experiments/credit/measure_connected_teaching_size_scale_v2.py','experiments/credit/connected_deep_teaching_actor.py','experiments/credit/check_native_packet_actor_update.py','experiments/credit/native_training_linearization.py','experiments/credit/native_cached_linearization.py','experiments/credit/native_cached_jvp.py','experiments/tpp/race_tpp_v19.py']
 result=dict(status='completed',tag=a.tag,battle='B1 exact actor plus forward-teaching accounting',rows=rows,source_sha256={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sources},data_sha256=gate['data_sha256'],wall_s=time.monotonic()-start,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scope='Real Taxi TRAIN prefixes, 8/32 larger native dense-write layers, complete forward/loss/backward/connected producer graph/1/16 input-direction teaching responses/Adam cost. Response buffers copied symmetrically. No critic fitting or downstream use included; this is the label-production boundary. Retained graph plus packet charged; no full-fit or quality/TEST claim.')
 with (ROOT/'experiments/results/credit'/f'{a.tag}.json').open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result),flush=True)
if __name__=='__main__':main()
