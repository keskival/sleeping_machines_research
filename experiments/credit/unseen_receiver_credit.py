"""B1/R1 shared critic for optional writes to never-labelled native receivers.
Frozen actor diagnostic; all receiver discovery and teacher work are charged.
"""
import argparse,hashlib,json,math,resource,time
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode
import structured_future_credit_v5 as prior
base=prior.base;ROOT=base.ROOT
SOURCES=['experiments/credit/unseen_receiver_credit.py',*prior.SOURCES]
CALLS=0
class RouteActor(base.ReceiverCuts):
    route_intervention=None
    def encode(self,t,m,mask):
        h,slots=super().encode(t,m,mask)
        if self.route_intervention is not None:
            index,receiver,delta=self.route_intervention;state=h.new_zeros(len(t),self.K,self.dk);history=[]
            for j in range(m.shape[1]):
                addressed=torch.nn.functional.one_hot(m[:,j],self.K).to(h.dtype)
                state=state*self.key_decay[:,j,None,:]+addressed[:,:,None]*self.written_keys[:,j,None,:]
                if j==index:state=state+torch.nn.functional.one_hot(receiver,self.K).to(h.dtype)[:,:,None]*delta[:,None,:]
                history.append(state)
            self._keyed=torch.stack(history,1)
        return h,slots

def actor(depth,seed):
    original=base.actor(depth,seed);model=RouteActor(8,8,3,depth,2,2,2,0.,1.,[-1.,0.],dk=4,local=True,prev_msg=True,qk_norm=False).double();model.load_state_dict(original.state_dict());return model

def losses(model,t,m,index=None,receiver=None,delta=None):
    global CALLS;CALLS+=1;model.route_intervention=None if index is None else (index,receiver,delta)
    try:return base.event_losses(model,t,m)
    finally:model.route_intervention=None

def prefix(model,t,m,index,receivers):
    global CALLS;CALLS+=1;x=base.prefix_packet(model,t,m,index);_,cache=losses(model,t[:,:index+1],m[:,:index+1]);params=model.clocks(cache[0],cache[1]);packets=[]
    for r in receivers:
        key=model._keyed[:,:,r];prob=params[4][...,r].exp().mean(-1,keepdim=True);match=(m[:,:index+1]==r).to(x.dtype).unsqueeze(-1);count=match.cumsum(1)/(torch.arange(index+1,dtype=x.dtype)[None,:,None]+1);role=torch.full_like(match,float(r>=4))
        packets.append(torch.cat((x,key,prob,match,count,role),-1))
    return torch.stack(packets,1).detach()

def responses(model,t,m,index,receivers,actions):
    _,cache=losses(model,t,m);query=model._query[:,:-1];age=(t[:,:-1]-t[:,index:index+1]).clamp_min(0)/model.scale;u=query*torch.exp(-age[:,:,None]*model.key_log_rate.exp())/math.sqrt(model.dk);lh,_=model.clock_terms(cache[3],*cache[2][:4]);target=cache[2][4].gather(-1,m[:,1:,None,None].expand(-1,-1,model.M,1)).squeeze(-1);logw=(lh+target).log_softmax(-1);values=[]
    for r in receivers:
        p=cache[2][4][...,r].exp();b=(m[:,1:]==r).to(query.dtype);values.append(prior.decode(u[:,index:],p[:,index:],logw[:,index:],b[:,index:],actions[None]).mean(1))
    return torch.stack(values,1)

def relabel(model,permutation):
    new=deepcopy(model);old=model.state_dict();state=new.state_dict();d=model.embed.embedding_dim;dv=model.marks.dv
    state['embed.weight'][permutation]=old['embed.weight'];state['mark_ctx.weight'].view(model.M,model.K,d)[:,permutation]=old['mark_ctx.weight'].view(model.M,model.K,d);state['mark_ctx.bias'].view(model.M,model.K)[:,permutation]=old['mark_ctx.bias'].view(model.M,model.K)
    state['clock.weight'][:,d:].view(-1,model.K,dv)[:,permutation]=old['clock.weight'][:,d:].reshape(-1,model.K,dv);new.load_state_dict(state);return new

def contracts():
    rows=[]
    for depth in (2,4,8):
        model=actor(depth,173);t,m=base.batch(173,size=1,pairs=2);actions=base.actions();index=3;exact=responses(model,t,m,index,list(range(8)),actions);original,_=losses(model,t,m);error=0.
        for receiver in range(8):
            action=actions[(receiver%8)+1][None];changed,_=losses(model,t,m,index,torch.tensor([receiver]),action);error=max(error,abs(float((changed[:,index:]-original[:,index:]).mean()-exact[0,receiver,(receiver%8)+1])));assert torch.equal(original[:,:index],changed[:,:index])
        assert error<1e-10,error
        delta=torch.zeros(1,4,dtype=torch.float64,requires_grad=True);changed,_=losses(model,t,m,index,torch.tensor([7]),delta);native_grad=torch.autograd.grad(changed[:,index:].mean(),delta)[0]
        d=delta.detach().clone().requires_grad_();g=torch.autograd.grad(responses(model,t,m,index,[7],d).mean(),d)[0];gradient_error=float((g-native_grad).abs().max());assert gradient_error<1e-10
        receivers=list(range(8));x=prefix(model,t,m,index,receivers);future=m.clone();future[:,index+1:]=(future[:,index+1:]+1)%8;causal=float((x-prefix(model,t,future,index,receivers)).abs().max());assert causal==0
        perm=torch.tensor([0,1,2,3,6,7,4,5]);new=relabel(model,perm);px=prefix(new,t,perm[m],index,perm.tolist());feature_error=float((x-px).abs().max());rx=responses(new,t,perm[m],index,perm.tolist(),actions);response_error=float((rx-exact).abs().max());assert feature_error<1e-10 and response_error<1e-10,(feature_error,response_error)
        rows.append(dict(depth=depth,optional_route_value_error=error,optional_route_gradient_error=gradient_error,causal_prefix_error=causal,value_relabel_descriptor_error=feature_error,value_relabel_response_error=response_error,receivers=8,prefix_feature_width=x.shape[-1]))
    return rows

def collect(model,seed,count,rollouts,receivers,dev=False):
    xs=[];ys=[];records=[]
    with torch.no_grad():
        for i in range(count):
            t,m=base.batch((2100000 if dev else 2000000)+seed*1000+i,size=1,pairs=2);xs.append(prefix(model,t,m,3,receivers));rng=np.random.default_rng((22000000 if dev else 21000000)+seed*10000+i);values=[]
            for j in range(rollouts):
                ft,fm=prior.continuation(t,m,2,rng);values.append(responses(model,ft,fm,3,receivers,base.actions())[0])
                if j==0:records.append((ft,fm,values[-1]))
            ys.append(torch.stack(values).mean(0))
    return torch.cat(xs),torch.stack(ys),records

def run(a):
    model=actor(a.depth,a.seed);U=prior.bound(model);x,y,train_records=collect(model,a.seed,a.train_packets,a.train_rollouts,list(range(6)));dx,dy,records=collect(model,a.seed,a.dev_packets,a.dev_rollouts,list(range(8)),True);scale=float(y.square().mean().sqrt().clamp_min(1e-8));width=x.shape[-1]
    torch.manual_seed(a.seed);structured=prior.KernelCritic(model.M,U).double();structured.encoder=nn.GRU(width,32,batch_first=True).double();head_width=round((structured.head.weight.numel()+structured.head.bias.numel()-1)/38);torch.manual_seed(a.seed);free=prior.FreeCritic(head_width,scale).double();free.encoder=nn.GRU(width,32,batch_first=True).double();free.encoder.load_state_dict(structured.encoder.state_dict());arms=[];saved={}
    aa=base.actions()[None].expand(len(x)*6,-1,-1);da=base.actions()[None].expand(len(dx)*8,-1,-1);xx=x.reshape(-1,x.shape[-2],width);yy=y.reshape(-1,9);dd=dx.reshape(-1,dx.shape[-2],width);fixed=int(y.mean(0).flatten().argmin());active=torch.tensor([int(m[0,3]) for _,m,_ in records]);rng=np.random.default_rng(a.seed);indices=rng.integers(0,len(dx),(2000,len(dx)));rolemean=torch.stack((y[:,:4].mean((0,1)),y[:,4:6].mean((0,1))));constant=rolemean[torch.tensor([0,0,0,0,1,1,1,1])][None].expand(len(dx),-1,-1)
    for name,critic in [('free',free),('structured',structured)]:
        initial=deepcopy(critic);opt=torch.optim.AdamW(critic.parameters(),lr=.003,weight_decay=.001)
        for step in range(a.steps):
            prediction=critic.values(xx,aa);loss=((prediction-yy)/scale).square().mean();opt.zero_grad();loss.backward();opt.step();assert torch.isfinite(loss)
        with torch.no_grad():
            pred=critic.values(dd,da).reshape(len(dx),8,9);frozen=initial.values(dd,da).reshape(len(dx),8,9);scopes={};route_indices=torch.arange(8)[None].expand(len(dx),-1)
            for scope,mask in [('seen',route_indices<6),('unseen',route_indices>=6),('unseen_inactive',(route_indices>=6)&(route_indices!=active[:,None]))]:
                truth=dy[mask];estimate=pred[mask];zero=truth.square().sum().clamp_min(1e-30);scopes[scope]=dict(receiver_packets=int(mask.sum()),relative_mse_vs_zero=float((estimate-truth).square().sum()/zero),frozen_relative_mse=float((frozen[mask]-truth).square().sum()/zero),role_constant_relative_mse=float((constant[mask]-truth).square().sum()/zero),mse_gain_vs_role_constant=float(((constant[mask]-truth).square()-(estimate-truth).square()).mean()))
            selected=pred.flatten(1).argmin(-1);restricted=pred[:,:6].flatten(1).argmin(-1);truth=dy.flatten(1);chosen=truth.gather(1,selected[:,None]).squeeze(1);pool=truth.gather(1,restricted[:,None]).squeeze(1);reference=truth[:,fixed];gain=(reference-chosen).numpy();expanded=(pool-chosen).numpy();receiver=selected//9;action=selected%9;outside=(receiver>=6)&(action!=0);inactive=outside&(receiver!=active);parity=0.
            for i,(ft,fm,first_truth) in enumerate(records[:8]):
                before,_=losses(model,ft,fm);changed,_=losses(model,ft,fm,3,receiver[i:i+1],base.actions()[action[i:i+1]]);parity=max(parity,abs(float((changed[:,3:]-before[:,3:]).mean()-first_truth.flatten()[selected[i]])));assert torch.equal(changed[:,:3],before[:,:3])
            assert parity<1e-10,parity
            utility=dict(gain_vs_train_fixed_action=float(gain.mean()),gain_vs_train_fixed_bootstrap95=np.quantile(gain[indices].mean(1),[.025,.975]).tolist(),gain_from_expanding_counterfactual_pool=float(expanded.mean()),pool_expansion_bootstrap95=np.quantile(expanded[indices].mean(1),[.025,.975]).tolist(),gain_vs_noop=float(-chosen.mean()),unseen_nonzero_selected=int(outside.sum()),unseen_inactive_selected=int(inactive.sum()),native_write_replay_error=parity,chosen_receivers=receiver.tolist(),chosen_action_indices=action.tolist(),chosen_future_advantages=chosen.tolist())
        arms.append(dict(arm=name,parameters=sum(p.numel() for p in critic.parameters()),scopes=scopes,utility=utility));saved[name]=critic.state_dict()
    checkpoint=ROOT/'experiments/results/credit'/f'{a.tag}.pt';assert not checkpoint.exists();torch.save(dict(actor=model.state_dict(),models=saved,args=vars(a),query_bound=U,feature_width=width,free_width=head_width,train_scale=scale),checkpoint)
    return dict(arms=arms,train_receivers=list(range(6)),unlabelled_receivers=[6,7],query_receivers=list(range(8)),actions=9,fixed_train_receiver=fixed//9,fixed_train_action=fixed%9,actor_forwards=CALLS,teacher_response_labels=a.train_packets*a.train_rollouts*6*9+a.dev_packets*a.dev_rollouts*8*9,receiver_descriptor_queries=a.train_packets*6+a.dev_packets*8,checkpoint=str(checkpoint.relative_to(ROOT)),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),scope='Frozen native actor; depth4 temporal/keyed memory and optional hard addressed key write. Receivers6/7 never receive TRAIN intervention labels; full route queries and response evaluation are paid. Fresh synthetic DEV continuations, no real stream/actor-weight fitting/internal route race/deployment efficiency claim. Both critics share labels/encoder initialization; nearly matched heads. Prefix bootstrap is conditional on fitted seed. Optional writes preserve temporal/value evolution by construction.')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);ap.add_argument('--mode',choices=['contract','smoke','pilot'],required=True);ap.add_argument('--seed',type=int,default=170);ap.add_argument('--depth',type=int,default=4);ap.add_argument('--steps',type=int,default=256);ap.add_argument('--train-packets',type=int,default=128);ap.add_argument('--dev-packets',type=int,default=64);ap.add_argument('--train-rollouts',type=int,default=8);ap.add_argument('--dev-rollouts',type=int,default=16);a=ap.parse_args();out=ROOT/'experiments/results/credit'/f'{a.tag}.json';assert not out.exists();torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
    if a.mode=='smoke':a.steps=2;a.train_packets=a.dev_packets=4;a.train_rollouts=a.dev_rollouts=2
    start=time.monotonic()
    with FlopCounterMode(display=False) as counter:metrics=contracts() if a.mode=='contract' else run(a)
    out.write_text(json.dumps(dict(status='completed',battle='B1/R1',tag=a.tag,args=vars(a),metrics=metrics,wall_s=time.monotonic()-start,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,supported_flops=counter.get_total_flops(),flop_scope='Teacher forward extraction, all receiver responses, both critic fits and native selected-write replays included; special/unsupported operations excluded.',source_sha256={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in SOURCES}),indent=2)+'\n')
if __name__=='__main__':main()
