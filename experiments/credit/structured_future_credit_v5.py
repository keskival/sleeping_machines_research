"""R1/B1 learned mixtures of exact local intervention-response kernels.
Frozen native actor; supervised counterfactual credit, not actor-training scaling.
"""
import argparse,hashlib,json,math,resource,time
from copy import deepcopy
import numpy as np
import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode
import future_route_critic_v1 as base
from conditional_credit_noise import continuation,analytic_advantages
SOURCES=['experiments/credit/structured_future_credit_v5.py','experiments/credit/conditional_credit_noise.py',*base.SOURCES]

def response(model,t,m,index):
    _,cache=base.event_losses(model,t,m);q=model._query[:,:-1]
    age=(t[:,:-1]-t[:,index:index+1]).clamp_min(0)/model.scale
    u=q*torch.exp(-age[:,:,None]*model.key_log_rate.exp())/math.sqrt(model.dk)
    logpk=cache[2][4];source=m[:,index]
    p=logpk.gather(-1,source[:,None,None,None].expand(-1,logpk.shape[1],model.M,1)).squeeze(-1).exp()
    target=logpk.gather(-1,m[:,1:,None,None].expand(-1,-1,model.M,1)).squeeze(-1)
    lh,_=model.clock_terms(cache[3],*cache[2][:4]);logw=(lh+target).log_softmax(-1)
    b=(m[:,1:]==source[:,None]).to(q.dtype)
    return tuple(v[:,index:] for v in (u,p,logw,b))

def decode(u,p,logw,b,actions):
    beta=torch.einsum('brd,bad->bra',u,actions)
    norm=torch.log1p(p[:,:,:,None]*torch.expm1(beta)[:,:,None,:])
    return -beta*b[:,:,None]+torch.logsumexp(logw,2)[:,:,None]-torch.logsumexp(logw[:,:,:,None]-norm,2)

def bound(model):
    norm=model.layers[-1].norm2
    hidden=norm.weight.abs().max()*math.sqrt(model.embed.embedding_dim)+norm.bias.norm()
    return float((torch.linalg.matrix_norm(model.query_w.weight,ord=2)*hidden+model.query_w.bias.norm())/math.sqrt(model.dk))

class KernelCritic(nn.Module):
    def __init__(self,M,U,R=8):
        super().__init__();self.M=M;self.U=U;self.R=R
        self.encoder=nn.GRU(30,32,batch_first=True);self.head=nn.Linear(32,R*(6+2*M))
        with torch.no_grad():self.head.weight.mul_(.02);self.head.bias.zero_()
    def values(self,x,actions):
        c=self.encoder(x.tanh())[0][:,-1];v=self.head(c).reshape(len(x),self.R,6+2*self.M)
        raw=v[:,:,:4];u=self.U*raw/(1+raw.norm(dim=-1,keepdim=True))
        p=v[:,:,4:4+self.M].sigmoid();logw=v[:,:,4+self.M:4+2*self.M].log_softmax(-1)
        b=v[:,:,-2].sigmoid();mix=v[:,:,-1].softmax(-1)
        return (decode(u,p,logw,b,actions)*mix[:,:,None]).sum(1)

class FreeCritic(nn.Module):
    def __init__(self,width,scale):
        super().__init__();self.encoder=nn.GRU(30,32,batch_first=True)
        self.head=nn.Sequential(nn.Linear(36,width),nn.Tanh(),nn.Linear(width,1));self.scale=scale
    def values(self,x,a):
        c=self.encoder(x.tanh())[0][:,-1,None,:].expand(-1,a.shape[1],-1)
        return self.scale*(self.head(torch.cat((c,a),-1))-self.head(torch.cat((c,torch.zeros_like(a)),-1))).squeeze(-1)

def contracts():
    rows=[]
    for depth in (2,4,8):
        model=base.actor(depth,181);t,m=base.batch(181,size=1,pairs=2);index=3
        gen=torch.Generator().manual_seed(181);extra=torch.randn(16,4,generator=gen,dtype=torch.float64);extra=.25*extra/extra.norm(dim=-1,keepdim=True)
        a=torch.cat((base.actions(),extra))[None];pack=response(model,t,m,index)
        values=decode(*pack,a).mean(1);reference=analytic_advantages(model,t,m,index,a[0]);error=float((values-reference).abs().max());assert error<1e-11
        delta=torch.zeros(1,1,4,dtype=torch.float64,requires_grad=True)
        grad=torch.autograd.grad(decode(*pack,delta).mean(),delta)[0]
        d=delta.detach().reshape(1,4).requires_grad_(True);loss,_=base.event_losses(model,t,m,index,d);g=torch.autograd.grad(loss[:,index:].mean(),d)[0]
        ge=float((grad.reshape(1,4)-g).abs().max());assert ge<1e-10
        critic=KernelCritic(model.M,bound(model)).double();x=base.prefix_packet(model,t,m,index)
        pred=critic.values(x,a);assert pred[0,0]==0 and pred.abs().max()<=critic.U*.25+1e-12
        rows.append(dict(depth=depth,value_parity_error=error,action_gradient_parity_error=ge,query_transport_bound=critic.U,candidates=25))
    return rows

def collect(model,seed,count,rollouts,actions,dev=False):
    xs=[];ys=[];examples=[]
    with torch.no_grad():
        for i in range(count):
            t,m=base.batch((1200000 if dev else 1100000)+seed*1000+i,size=1,pairs=2);xs.append(base.prefix_packet(model,t,m,3))
            rng=np.random.default_rng((15000000 if dev else 14000000)+seed*10000+i);values=[]
            for k in range(rollouts):
                ft,fm=continuation(t,m,2,rng);values.append(decode(*response(model,ft,fm,3),actions[None]).mean(1)[0])
                if k==0:examples.append((ft,fm,values[-1]))
            ys.append(torch.stack(values).mean(0))
    return torch.cat(xs),torch.stack(ys),examples

def run(a):
    model=base.actor(a.depth,a.seed);U=bound(model);gen=torch.Generator().manual_seed(181)
    extra=torch.randn(16,4,generator=gen,dtype=torch.float64);extra=.25*extra/extra.norm(dim=-1,keepdim=True)
    candidates=torch.cat((base.actions(),extra));x,y,records=collect(model,a.seed,a.train_packets,a.train_rollouts,candidates)
    dx,dy,examples=collect(model,a.seed,a.dev_packets,a.dev_rollouts,candidates,True)
    scale=float(y[:,:9].square().mean().sqrt().clamp_min(1e-8));torch.manual_seed(a.seed);structured=KernelCritic(model.M,U).double()
    width=round((structured.head.weight.numel()+structured.head.bias.numel()-1)/38);torch.manual_seed(a.seed);free=FreeCritic(width,scale).double();free.encoder.load_state_dict(structured.encoder.state_dict())
    arms=[];saved={};rng=np.random.default_rng(a.seed);indices=rng.integers(0,len(dx),(2000,len(dx)))
    for name,critic in [('free',free),('structured',structured)]:
        opt=torch.optim.AdamW(critic.parameters(),lr=.003,weight_decay=.001)
        aa=candidates[None,:9].expand(len(x),-1,-1)
        for k in range(a.steps):
            loss=((critic.values(x,aa)-y[:,:9])/scale).square().mean();opt.zero_grad();loss.backward();opt.step();assert torch.isfinite(loss)
        scopes={}
        with torch.no_grad():
            pred=critic.values(dx,candidates[None].expand(len(dx),-1,-1))
            for scope,sl in [('basis',slice(0,9)),('unseen_directions',slice(9,25)),('augmented_pool',slice(0,25))]:
                truth=dy[:,sl];estimate=pred[:,sl];constant=y[:,sl].mean(0);fixed=int(constant.argmin());selected=estimate.argmin(-1);chosen=truth.gather(1,selected[:,None]).squeeze(1)
                errors=((estimate-truth)**2).mean(-1);msegain=(((truth-constant)**2).mean(-1)-errors).numpy();utility=(truth[:,fixed]-chosen).numpy()
                scopes[scope]=dict(relative_mse=float((estimate-truth).square().sum()/truth.square().sum().clamp_min(1e-30)),mse_gain_vs_train_constant=float(msegain.mean()),utility_gain_vs_train_constant=float(utility.mean()),utility_gain_bootstrap95=np.quantile(utility[indices].mean(1),[.025,.975]).tolist(),mean_pool_regret=float((chosen-truth.min(-1).values).mean()),chosen_advantages=chosen.tolist())
            chosen=pred.argmin(-1);parity=0.
            for i,(ft,fm,exact) in enumerate(examples[:4]):
                action=candidates[chosen[i]];original,_=base.event_losses(model,ft,fm);changed,_=base.event_losses(model,ft,fm,3,action[None])
                parity=max(parity,abs(float((changed[:,3:]-original[:,3:]).mean()-exact[chosen[i]])));assert torch.equal(changed[:,:3],original[:,:3])
            assert parity<1e-11
        arms.append(dict(arm=name,parameters=sum(p.numel() for p in critic.parameters()),scopes=scopes,chosen_native_write_parity_error=parity));saved[name]=critic.state_dict()
    checkpoint=base.ROOT/'experiments/results/credit'/f'{a.tag}.pt';assert not checkpoint.exists();torch.save(dict(models=saved,actor_state=model.state_dict(),args=vars(a),query_bound=U,free_width=width,train_scale=scale),checkpoint)
    teacher_calls=a.train_packets*(a.train_rollouts+1)+a.dev_packets*(a.dev_rollouts+1)+16
    return dict(arms=arms,actor_depth=a.depth,query_bound=U,components=8,critic_training_actions=9,evaluation_actions=25,checkpoint=str(checkpoint.relative_to(base.ROOT)),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),teacher_actor_forwards=teacher_calls,teacher_target_presentations=a.train_packets*(7*a.train_rollouts+3)+a.dev_packets*(7*a.dev_rollouts+3)+112,scope='Matched prefix encoders, head parameter counts differ by rounding only, common basis TRAIN labels/scale and fit steps. Sixteen unseen directions never enter critic fitting. TRAIN constant for unseen directions is computed from the same TRAIN continuations, giving the reference extra action information. Selected native writes checked on four independent DEV prefixes, all replay charged. Initialized frozen teacher, synthetic true-generator access, no actor weight fitting, online scheduling or hardware efficiency claim.')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);ap.add_argument('--mode',choices=['contract','smoke','pilot'],required=True);ap.add_argument('--depth',type=int,default=4);ap.add_argument('--seed',type=int,default=170);ap.add_argument('--steps',type=int,default=256);ap.add_argument('--train-packets',type=int,default=128);ap.add_argument('--dev-packets',type=int,default=64);ap.add_argument('--train-rollouts',type=int,default=8);ap.add_argument('--dev-rollouts',type=int,default=16);a=ap.parse_args()
    if a.mode=='smoke':a.steps=2;a.train_packets=a.dev_packets=4;a.train_rollouts=a.dev_rollouts=2
    output=base.ROOT/'experiments/results/credit'/f'{a.tag}.json';assert not output.exists();torch.set_num_threads(1);torch.set_default_dtype(torch.float64);start=time.monotonic()
    with FlopCounterMode(display=False) as counter:metrics=contracts() if a.mode=='contract' else run(a)
    result=dict(status='completed',tag=a.tag,battle='R1/B1',args=vars(a),metrics=metrics,wall_s=time.monotonic()-start,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,supported_flops=counter.get_total_flops(),flop_scope='All teacher response extraction, both fits, 25-action evaluation and selected-write replays included; special/unsupported arithmetic excluded.',source_sha256={p:hashlib.sha256((base.ROOT/p).read_bytes()).hexdigest() for p in SOURCES})
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='metrics'}),flush=True)
if __name__=='__main__':main()
