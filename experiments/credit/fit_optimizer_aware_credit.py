"""B1/R1 equal-capacity fixed-path vs future-learning-aware correction heads."""
import argparse,hashlib,json,resource,time
from copy import deepcopy
import numpy as np
import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode
from optimizer_aware_route_credit import replay,native,ROOT
from shared_prefix_receiver_credit import SharedPrefixCredit
class Correction(nn.Module):
    def __init__(self,scale):
        super().__init__();self.scale=scale;self.net=nn.Sequential(nn.Linear(42,64),nn.Tanh(),nn.Linear(64,1));nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def values(self,z,a):
        x=z[:,:,None,:].expand(-1,-1,a.shape[2],-1);return self.scale*(self.net(torch.cat((x,a),-1))-self.net(torch.cat((x,torch.zeros_like(a)),-1))).squeeze(-1)
def collect(model,critic,seed,count,rollouts,routes,lr,dev):
    features=[];base=[];static=[];aware=[]
    for i in range(count):
        t,m=native.base.batch((4100000 if dev else 4000000)+seed*1000+i,size=1,pairs=2)
        with torch.no_grad():
            x=native.prefix(model,t,m,3,routes);context=critic.encode_context(x[:,0,:,:30],0,0);r=critic.receiver(x[:,:,:,30:].reshape(len(routes),4,8).tanh())[0][:,-1].reshape(1,len(routes),7);z=torch.cat((context.value[:,None].expand(-1,len(routes),-1),r),-1);a=native.base.actions()[None,None].expand(1,len(routes),-1,-1);base.append(critic.values(context,x[:,:,:,30:],a,0,0)[0]);features.append(z[0])
        rng=np.random.default_rng((43000000 if dev else 42000000)+seed*1000+i);frozen=[];learning=[]
        for j in range(rollouts):
            ft,fm=native.prior.continuation(t,m,2,rng)
            with torch.no_grad():frozen.append(native.responses(model,ft,fm,3,routes,native.base.actions())[0])
            scores,_,_=replay(model,ft,fm,routes,native.base.actions(),lr);value=scores.mean(1).reshape(len(routes),9);learning.append(value-value[:,:1])
        static.append(torch.stack(frozen).mean(0));aware.append(torch.stack(learning).mean(0))
    return tuple(torch.stack(xs).detach() for xs in [features,base,static,aware])
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);ap.add_argument('--mode',choices=['smoke','pilot'],required=True);a=ap.parse_args();torch.set_num_threads(1);torch.set_default_dtype(torch.float64);out=ROOT/'experiments/results/credit'/f'{a.tag}.json';assert not out.exists();start=time.monotonic();source=ROOT/'experiments/results/credit/curie_shared_receiver_fit_20261010T2300Z_pilot.json';old=json.loads(source.read_text());cp=ROOT/old['checkpoint'];assert hashlib.sha256(cp.read_bytes()).hexdigest()==old['checkpoint_sha256'];saved=torch.load(cp,map_location='cpu',weights_only=False);model=native.actor(4,170);model.load_state_dict(saved['actor']);critic=SharedPrefixCredit(model.M,native.prior.bound(model)).double();critic.load_state_dict(saved['critic']);critic.requires_grad_(False);small=a.mode=='smoke';n,nd,nt,nr,steps=(2,2,1,2,2) if small else (32,16,4,8,256);lr=.05
    with FlopCounterMode(display=False) as teacher:
        z,b,ys,ya=collect(model,critic,170,n,nt,list(range(6)),lr,False);dz,db,ds,da=collect(model,critic,170,nd,nr,list(range(8)),lr,True)
    scale=float(ys.square().mean().sqrt().clamp_min(1e-8));torch.manual_seed(170);first=Correction(scale).double();heads={'fixed_path_fit':first,'learning_aware_fit':deepcopy(first)};action=native.base.actions()[None,None].expand(n,6,-1,-1);dev_action=native.base.actions()[None,None].expand(nd,8,-1,-1);fixed=int(ya.mean(0).flatten().argmin());indices=np.random.default_rng(170).integers(0,nd,(2000,nd));rows=[];saved_heads={};saved_opts={};fit_work=[]
    for name,head in heads.items():
        target=ys if name=='fixed_path_fit' else ya;opt=torch.optim.AdamW(head.parameters(),lr=.003,weight_decay=.001);tick=time.monotonic()
        with FlopCounterMode(display=False) as counter:
            for step in range(steps):
                prediction=b+head.values(z,action);loss=((prediction-target)/scale).square().mean();assert torch.isfinite(loss);opt.zero_grad();loss.backward();opt.step()
        fit_work.append(dict(arm=name,wall_s=time.monotonic()-tick,supported_flops=counter.get_total_flops()));saved_heads[name]=head.state_dict();saved_opts[name]=opt.state_dict()
    predictions={'saved_frozen_credit':db}
    with torch.no_grad():
        for name,head in heads.items():predictions[name]=db+head.values(dz,dev_action)
        selected_values={}
        for name,pred in predictions.items():
            chosen=pred.flatten(1).argmin(-1);truth=da.flatten(1);value=truth.gather(1,chosen[:,None]).squeeze(1);selected_values[name]=value;gain=(truth[:,fixed]-value).numpy();restriction=pred[:,:6].flatten(1).argmin(-1);extra=(truth.gather(1,restriction[:,None]).squeeze(1)-value).numpy()
            rows.append(dict(arm=name,learning_aware_relative_mse=float((pred-da).square().sum()/da.square().sum()),unseen_learning_aware_relative_mse=float((pred[:,6:]-da[:,6:]).square().sum()/da[:,6:].square().sum()),gain_vs_train_fixed=float(gain.mean()),gain_bootstrap95=np.quantile(gain[indices].mean(1),[.025,.975]).tolist(),gain_vs_noop=float(-value.mean()),expanded_pool_gain=float(extra.mean()),unseen_nonzero_selected=int(((chosen//9>=6)&(chosen%9!=0)).sum()),chosen_receiver_action=torch.stack((chosen//9,chosen%9),1).tolist()))
        diff=(selected_values['fixed_path_fit']-selected_values['learning_aware_fit']).numpy();paired=dict(gain_learning_aware_over_fixed_path=float(diff.mean()),bootstrap95=np.quantile(diff[indices].mean(1),[.025,.975]).tolist())
    checkpoint=out.with_suffix('.pt');torch.save(dict(actor=model.state_dict(),baseline=critic.state_dict(),heads=saved_heads,optimizers=saved_opts,train_tensors=dict(z=z,base=b,fixed=ys,aware=ya),dev_tensors=dict(z=dz,base=db,fixed=ds,aware=da),args=vars(a),readout_adam_lr=lr),checkpoint)
    sources=['experiments/credit/fit_optimizer_aware_credit.py','experiments/credit/optimizer_aware_route_credit.py','experiments/credit/shared_prefix_receiver_credit.py',*native.SOURCES]
    result=dict(status='completed',battle='B1/R1',tag=a.tag,args=vars(a),rows=rows,paired=paired,teacher_supported_flops=teacher.get_total_flops(),fit_work=fit_work,train_prefixes=n,dev_prefixes=nd,train_rollouts=nt,dev_rollouts=nr,steps=steps,readout_adam_lr=lr,correction_parameters=sum(p.numel() for p in first.parameters()),fixed_response_vs_learning_response_mse=float((ds-da).square().mean()),checkpoint=str(checkpoint.relative_to(ROOT)),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),prior_result=str(source.relative_to(ROOT)),prior_checkpoint_sha256=old['checkpoint_sha256'],whole_wall_s=time.monotonic()-start,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},scope='Seed170 bounded synthetic development. Branch-local288-parameter readout Adam adapts after each of four revealed targets; carrier and critic remain fixed during each continuation. Full-prefix causal replay and all baseline/teacher/fitting work charged. Identical correction heads/initialization/steps/observations; labels deliberately compare frozen response with actual future-learning response. TRAIN labels only receivers0–5. DEV continuations are independent; intervals bootstrap prefixes conditional on fitted seed. No deep actor co-adaptation, online critic learning, general internal races, matched-quality whole-fit efficiency, hardware or TEST claim.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(rows=rows,paired=paired),indent=2))
if __name__=='__main__':main()
