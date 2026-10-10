"""B1/R1 shared context and receiver-conditioned structured credit; contracts."""
import argparse,hashlib,json,resource,time
from dataclasses import dataclass
import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode
import structured_future_credit_v5 as prior
ROOT=prior.base.ROOT
@dataclass
class ContextActivation:
    value:torch.Tensor
    forward_version:int
    critic_version:int
class SharedPrefixCredit(nn.Module):
    def __init__(self,M,U,R=8):
        super().__init__();self.M=M;self.U=U;self.R=R
        self.context=nn.GRU(30,31,batch_first=True);self.receiver=nn.GRU(8,7,batch_first=True);self.head=nn.Linear(38,R*(6+2*M))
        with torch.no_grad():self.head.weight.mul_(.02);self.head.bias.zero_()
    def encode_context(self,x,forward_version,critic_version):
        return ContextActivation(self.context(x.tanh())[0][:,-1],forward_version,critic_version)
    def values(self,activation,routes,actions,forward_version,critic_version):
        if (activation.forward_version,activation.critic_version)!=(forward_version,critic_version):raise ValueError('Stale context activation')
        B,K,T,W=routes.shape;assert W==8 and actions.shape[:2]==(B,K)
        r=self.receiver(routes.reshape(B*K,T,8).tanh())[0][:,-1];c=activation.value[:,None,:].expand(-1,K,-1).reshape(B*K,31)
        v=self.head(torch.cat((c,r),-1)).reshape(B*K,self.R,6+2*self.M);raw=v[:,:,:4];u=self.U*raw/(1+raw.norm(dim=-1,keepdim=True));p=v[:,:,4:4+self.M].sigmoid();logw=v[:,:,4+self.M:4+2*self.M].log_softmax(-1);b=v[:,:,-2].sigmoid();mix=v[:,:,-1].softmax(-1)
        return (prior.decode(u,p,logw,b,actions.reshape(B*K,-1,4))*mix[:,:,None]).sum(1).reshape(B,K,-1)
def contracts():
    torch.manual_seed(184);model=SharedPrefixCredit(4,2.).double();x=torch.randn(2,4,30);r=torch.randn(2,8,4,8);actions=prior.base.actions()[None,None].expand(2,8,-1,-1);v=model.encode_context(x,5,7);all_values=model.values(v,r,actions,5,7);pick=torch.tensor([2,5]);selected=model.values(v,r[:,pick],actions[:,pick],5,7);subset_error=float((selected-all_values[:,pick]).abs().max());assert subset_error<1e-12
    permutation=torch.tensor([6,1,7,0,4,3,2,5]);permuted=model.values(v,r[:,permutation],actions[:,permutation],5,7);permutation_error=float((permuted-all_values[:,permutation]).abs().max());assert permutation_error<1e-12
    altered=r.clone();altered[:,0]=100.;untouched=model.values(v,altered[:,pick],actions[:,pick],5,7);unqueried_error=float((selected-untouched).abs().max());assert unqueried_error==0
    g1=torch.autograd.grad(selected.square().sum(),tuple(model.parameters()));repeated=[]
    for k in pick.tolist():
        cv=model.encode_context(x,5,7);repeated.append(model.values(cv,r[:,k:k+1],actions[:,k:k+1],5,7))
    g2=torch.autograd.grad(torch.cat(repeated,1).square().sum(),tuple(model.parameters()));gradient_error=max(float((a-b).abs().max()) for a,b in zip(g1,g2));assert gradient_error<1e-12
    rejected=0
    for fv,cv in [(6,7),(5,8)]:
        try:model.values(v,r[:,pick],actions[:,pick],fv,cv)
        except ValueError:rejected+=1
    assert rejected==2 and float(all_values[:,:,0].abs().max())==0
    counts=[]
    with torch.no_grad():
        for budget in [1,2,4,8]:
            with FlopCounterMode(display=False) as counter:
                cv=model.encode_context(x,5,7);model.values(cv,r[:,:budget],actions[:,:budget],5,7)
            counts.append(dict(receiver_budget=budget,supported_flops=counter.get_total_flops()))
    return dict(parameters=sum(p.numel() for p in model.parameters()),subset_value_error=subset_error,receiver_permutation_error=permutation_error,unqueried_receiver_error=unqueried_error,shared_context_gradient_error=gradient_error,stale_version_rejections=rejected,no_op_error=0.,query_work=counts,retained_context_output_values=int(v.value.numel()),scope='Random circuit contracts, not learned accuracy. Context output count excludes fitting graphs, weights, optimizer, packets and receiver state. Supported FLOPs exclude special/unsupported operators. Changed critic representation; native forward actor retained for subsequent fitted comparison.')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args();torch.set_num_threads(1);torch.set_default_dtype(torch.float64);out=ROOT/'experiments/results/credit'/f'{a.tag}.json';assert not out.exists();start=time.monotonic();metrics=contracts();sources=['experiments/credit/shared_prefix_receiver_credit.py',*prior.SOURCES]
    out.write_text(json.dumps(dict(status='completed',battle='B1/R1',tag=a.tag,metrics=metrics,wall_s=time.monotonic()-start,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,source_sha256={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sources}),indent=2)+'\n');print(json.dumps(metrics,indent=2))
if __name__=='__main__':main()
