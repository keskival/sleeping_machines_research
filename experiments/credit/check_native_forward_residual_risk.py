"""B1 native observation-loss cotangent risk labels, without reverse teaching."""
import argparse,json,time,hashlib,resource,sys,random
from pathlib import Path
import torch,numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'experiments/tpp'))
from check_native_packet_actor_update import native
from connected_teaching_actor import ConnectedTeachingActor
from forward_residual_risk import residual_observations
from predictable_audit_allocation_v2 import ResidualRiskLearner

def orthogonal_probes(dimension):
 h=torch.ones(1,1)
 while len(h)<dimension:h=torch.cat([torch.cat([h,h],1),torch.cat([h,-h],1)],0)
 return h[:,:dimension]
def features(model,t):
 dt=torch.diff(t,dim=1,prepend=t[:,:1]).clamp_min(0)/model.scale;age=(t-t[:,:1])/model.scale
 return torch.stack([torch.log1p(dt),(dt>0).to(t.dtype),torch.log1p(age)],-1)
def producer_directions(model,m,f,v):
 e=model.embed.weight.numel();w=model.gap.weight.numel();d=model.d if hasattr(model,'d') else model.embed.weight.shape[1]
 embedding=v[:,:e].reshape(len(v),*model.embed.weight.shape);weight=v[:,e:e+w].reshape(len(v),*model.gap.weight.shape);bias=v[:,e+w:].reshape(len(v),d)
 return embedding[:,m]+torch.einsum('blf,qdf->qbld',f,weight)+bias[:,None,None,:]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args();torch.set_num_threads(1);torch.set_default_dtype(torch.float64);torch.manual_seed(711);start=time.monotonic()
 train=native.load_split('taxi','train');seq=next(x for x in train if len(x[1])>=4);K=1+max(int(m.max()) for _,m in train);positive=np.diff(seq[0][:4]);positive=positive[positive>0];scale=float(np.median(positive));qs=np.log(np.quantile(positive,[.1,.5,.9])).tolist()
 model=ConnectedTeachingActor(K,8,4,1,2,3,2,0.,scale,qs).eval();model.credit_version=0;t,m,mask=next(native.batches([(seq[0][:4],seq[1][:4])],1,False,random.Random(0)))
 tl,ml,count,_=model.loglik(t,m,mask);h=model.packet.forward_output;h.retain_grad();loss=-(tl+ml)/count;loss.backward();c=h.grad.detach();predicted=.5*c
 directions=orthogonal_probes(c.numel()).reshape(-1,*c.shape);f=features(model,t);P=sum(p.numel() for p in [model.embed.weight,model.gap.weight,model.gap.bias]);parameter_probes=orthogonal_probes(P);pd=producer_directions(model,m,f,parameter_probes)
 tick=time.monotonic();response=model.packet.apply(directions,context='taxi',version=0);delta,labels=residual_observations(c,predicted,directions,response);pres=model.packet.apply(pd,context='taxi',version=0);pdelta,plabels=residual_observations(c,predicted,pd,pres);label_s=time.monotonic()-tick
 # Reverse maps are validation instruments only; not used to construct labels.
 tick=time.monotonic();exact=model.packet.transpose(c[None],context='taxi',version=0)[0];r=exact-predicted;direct= float(r.square().sum());input_error=abs(float(labels.mean())-direct)
 x=model.embed(m)+model.gap(f);pg=torch.autograd.grad((x*r).sum(),[model.embed.weight,model.gap.weight,model.gap.bias]);producer_exact=sum(float(g.square().sum()) for g in pg);producer_error=abs(float(plabels.mean())-producer_exact);validation_s=time.monotonic()-tick
 assert input_error<1e-10 and producer_error<1e-10
 risk=ResidualRiskLearner(2);issued=risk.issue([1.,.5],1.);gradient=np.mean([risk.gradient(issued,audited=True,residual_squared=float(z)) for z in labels],axis=0);reference=risk.gradient(issued,audited=True,residual_squared=direct);risk_gradient_error=float(np.max(np.abs(gradient-reference)));assert risk_gradient_error<1e-10
 sources=['experiments/credit/check_native_forward_residual_risk.py','experiments/credit/forward_residual_risk.py','experiments/credit/connected_teaching_actor.py','experiments/credit/check_native_packet_actor_update.py','experiments/credit/native_training_linearization.py','experiments/credit/native_cached_linearization.py','experiments/credit/native_cached_jvp.py','experiments/credit/predictable_audit_allocation_v2.py','experiments/tpp/race_tpp_v19.py']
 result=dict(status='completed',tag=a.tag,battle='B1 forward-derived credit risk',loss=float(loss.detach()),input_risk=direct,input_risk_error=input_error,producer_risk=producer_exact,producer_risk_error=producer_error,risk_training_gradient_error=risk_gradient_error,input_probe_count=len(directions),producer_probe_count=len(pd),producer_parameter_count=P,label_s=label_s,reverse_validation_s=validation_s,source_sha256={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sources},data_sha256=hashlib.sha256((ROOT/'data/easytpp/taxi/train.json').read_bytes()).hexdigest(),wall_s=time.monotonic()-start,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scope='One four-event Taxi TRAIN prefix/one native layer, actual observed-loss cotangent. Exhaustive orthogonal directions establish unbiased cut/producer risk and conditional risk-training gradient; labels use forward responses only. Reverse validation paid separately. No learned-credit performance, sparse sampling, quality, meta-optimizer or full-learning efficiency claim.')
 with (ROOT/'experiments/results/credit'/f'{a.tag}.json').open('x') as out:json.dump(result,out,indent=2)
 print(json.dumps(result))
if __name__=='__main__':main()
