"""B1 integrated actor-gradient/Adam contract for shared cached primal packet."""
import argparse,hashlib,json,resource,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'experiments/tpp'))
import race_tpp_v19 as native
from native_training_linearization import NativeTrainingLinearization

class PacketActor(native.RaceTPP):
 def encode(self,t,m,mask):
  dt=torch.diff(t,dim=1,prepend=t[:,:1]).clamp_min(0)/self.scale;age=(t-t[:,:1])/self.scale
  x=self.embed(m)+self.gap(torch.stack([torch.log1p(dt),(dt>0).to(t.dtype),torch.log1p(age)],-1))
  self.cut_input=x
  assert len(self.layers)==1
  self.packet=NativeTrainingLinearization.capture(self.layers[0],x.detach(),dt,context='taxi',version=self.credit_version)
  h=self.packet.forward_output;h.retain_grad();self.cut_output=h
  return h,self.marks(h,m,dt)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args()
 torch.set_num_threads(1);torch.set_default_dtype(torch.float64);torch.manual_seed(922);start=time.monotonic()
 train=native.load_split('taxi','train');seq=next(s for s in train if len(s[1])>=4);K=1+max(int(m.max()) for _,m in train)
 pos=np.diff(seq[0][:4]);pos=pos[pos>0];scale=float(np.median(pos));qs=np.log(np.quantile(pos,[.1,.5,.9])).tolist()
 args=(K,8,4,1,2,3,2,0.,scale,qs);ref=native.RaceTPP(*args).eval();actor=PacketActor(*args).eval();actor.load_state_dict(ref.state_dict())
 ropt=torch.optim.Adam(ref.parameters(),lr=.001);opt=torch.optim.Adam(actor.parameters(),lr=.001)
 t=torch.tensor(seq[0][:4])[None];m=torch.tensor(seq[1][:4])[None];mask=torch.ones_like(m,dtype=torch.bool);history=[]
 for version in range(4):
  actor.credit_version=version;ropt.zero_grad();opt.zero_grad()
  tl,ml,count,_=ref.loglik(t,m,mask);reference=-(tl+ml)/count;reference.backward()
  tl,ml,count,_=actor.loglik(t,m,mask);loss=-(tl+ml)/count;loss.backward()
  carried=actor.packet.transpose(actor.cut_output.grad[None],context='taxi',version=version)[0]
  actor.cut_input.backward(carried)
  error=max(float((p.grad-q.grad).abs().max())/(1+float(p.grad.abs().max())) for p,q in zip(ref.parameters(),actor.parameters()))
  assert error<1e-10 and abs(float(loss-reference))<1e-10
  ropt.step();opt.step();param=max(float((p-q).detach().abs().max()) for p,q in zip(ref.parameters(),actor.parameters()))
  adam=max(float((ropt.state[p][k]-opt.state[q][k]).abs().max()) for p,q in zip(ref.parameters(),actor.parameters()) for k in ['exp_avg','exp_avg_sq']);assert param<1e-9 and adam<1e-9
  try:actor.packet.transpose(actor.cut_output.grad[None],context='taxi',version=version)
  except ValueError:pass
  else:raise AssertionError('packet survived parameter update')
  history.append(dict(version=version,loss=float(loss.detach()),gradient_error=error,parameter_error=param,adam_error=adam,cached_floats=actor.packet.cached_floats()))
 sources=['experiments/credit/check_native_packet_actor_update.py','experiments/credit/native_training_linearization.py','experiments/credit/native_cached_linearization.py','experiments/credit/native_cached_jvp.py','experiments/tpp/race_tpp_v19.py']
 result=dict(status='completed',tag=a.tag,battle='B1 integrated exact packet credit',history=history,source_sha256={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sources},data_sha256=hashlib.sha256((ROOT/'data/easytpp/taxi/train.json').read_bytes()).hexdigest(),wall_s=time.monotonic()-start,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scope='One four-event real Taxi TRAIN prefix, one native layer,4 Adam updates. One tracked forward capture retains exact layer/head parameter graph; packet VJP reconnects detached embedding/gap producers. No dense Jacobian/probe sweep. All actor gradients/Adam checked. Graph plus packet storage remains charged; no whole-fit speed, learned-credit generalization, TEST or quality claim.')
 with (ROOT/'experiments/results/credit'/f'{a.tag}.json').open('x') as out:out.write(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result),flush=True)
if __name__=='__main__':main()
