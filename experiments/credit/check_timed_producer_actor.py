"""B1 exact joint packet reconnects content and learnable processing-time producers."""
import argparse,json,time,hashlib,resource,sys,math
from pathlib import Path
import torch,numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'experiments/tpp'))
from check_native_packet_actor_update import native
from native_training_temporal_packet import NativeTrainingTemporalPacket
class TimedActor(native.RaceTPP):
 def __init__(self,*args):
  super().__init__(*args);self.tempo=torch.nn.Linear(3,1)
  with torch.no_grad():self.tempo.weight.zero_();self.tempo.bias.fill_(math.log(math.expm1(1.)))
 def encode(self,t,m,mask):
  dt=torch.diff(t,dim=1,prepend=t[:,:1]).clamp_min(0)/self.scale;age=(t-t[:,:1])/self.scale;f=torch.stack([torch.log1p(dt),(dt>0).to(t.dtype),torch.log1p(age)],-1)
  x=self.embed(m)+self.gap(f);elapsed=dt*torch.nn.functional.softplus(self.tempo(f).squeeze(-1));assert len(self.layers)==1
  if getattr(self,'packet_mode',False):
   self.cut_input=x;self.cut_elapsed=elapsed;self.packet=NativeTrainingTemporalPacket.capture(self.layers[0],x.detach(),elapsed.detach(),context='taxi',version=self.credit_version);h=self.packet.forward_output;h.retain_grad();self.cut_output=h
  else:h=self.layers[0](x,elapsed)
  # Addressed mark memory and race density retain physical observed time.
  return h,self.marks(h,m,dt)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args();torch.set_num_threads(1);torch.set_default_dtype(torch.float64);torch.manual_seed(942);start=time.monotonic();train=native.load_split('taxi','train');seq=next(s for s in train if len(s[1])>=4);K=1+max(int(m.max()) for t,m in train);pos=np.diff(seq[0][:4]);pos=pos[pos>0];scale=float(np.median(pos));qs=np.log(np.quantile(pos,[.1,.5,.9])).tolist();args=(K,8,4,1,2,3,2,0.,scale,qs)
 ref=TimedActor(*args).eval();actor=TimedActor(*args).eval();actor.load_state_dict(ref.state_dict());actor.packet_mode=True;ropt=torch.optim.Adam(ref.parameters(),lr=.001);opt=torch.optim.Adam(actor.parameters(),lr=.001);t=torch.tensor(seq[0][:4])[None];m=torch.tensor(seq[1][:4])[None];mask=torch.ones_like(m,dtype=torch.bool);history=[]
 for version in range(4):
  actor.credit_version=version;ropt.zero_grad();opt.zero_grad();tl,ml,n,_=ref.loglik(t,m,mask);rloss=-(tl+ml)/n;rloss.backward();tl,ml,n,_=actor.loglik(t,m,mask);loss=-(tl+ml)/n;loss.backward();gx,gt=actor.packet.transpose_with_time(actor.cut_output.grad[None],context='taxi',version=version);torch.autograd.backward([actor.cut_input,actor.cut_elapsed],[gx[0],gt[0]])
  gradients=max(float((p.grad-q.grad).abs().max()) for p,q in zip(ref.parameters(),actor.parameters()));tempo_gradient=float(actor.tempo.weight.grad.abs().sum()+actor.tempo.bias.grad.abs().sum());assert gradients<1e-10 and tempo_gradient>0 and abs(float(loss-rloss))<1e-10
  ropt.step();opt.step();parameters=max(float((p-q).detach().abs().max()) for p,q in zip(ref.parameters(),actor.parameters()));adam=max(float((ropt.state[p][k]-opt.state[q][k]).abs().max()) for p,q in zip(ref.parameters(),actor.parameters()) for k in ['exp_avg','exp_avg_sq']);assert parameters<1e-10 and adam<1e-10
  history.append(dict(version=version,loss=float(loss.detach()),gradient_error=gradients,parameter_error=parameters,adam_error=adam,tempo_gradient_norm=tempo_gradient))
 sources=['experiments/credit/check_timed_producer_actor.py','experiments/credit/native_training_temporal_packet.py','experiments/credit/native_temporal_teaching_packet.py','experiments/credit/native_training_linearization.py','experiments/credit/native_cached_linearization.py','experiments/credit/native_cached_jvp.py','experiments/tpp/race_tpp_v19.py'];r=dict(status='completed',tag=a.tag,battle='B1 joint temporal-credit execution',history=history,source_sha256={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sources},data_sha256=hashlib.sha256((ROOT/'data/easytpp/taxi/train.json').read_bytes()).hexdigest(),wall_s=time.monotonic()-start,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scope='One4-event TRAIN prefix/4Adam updates, diagnostic causal learned positive internal tempo, physical mark/race time unchanged. Exact joint packet producer-credit parity, native layer/head parameter graph retained; no learned backward estimator, quality/depth/speed/TEST claim or active architecture substitution.')
 with (ROOT/'experiments/results/credit'/f'{a.tag}.json').open('x') as out:json.dump(r,out,indent=2)
 print(json.dumps(r))
if __name__=='__main__':main()
