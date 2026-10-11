"""R1 whole learner-state derivative: eligibility, reverse graph and finite differences."""
import argparse,hashlib,json,time
import torch
from torch.func import functional_call
from check_token_forward_credit import ROOT
from online_geometric_update import UpdateFunction
from multistep_geometric_update import future_step

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args();out=ROOT/f'experiments/results/credit/{a.tag}.json';assert not out.exists();start=time.monotonic();torch.set_num_threads(1);torch.set_default_dtype(torch.float64);checks=[]
    for magnitude in [1.,.1]:
        torch.manual_seed(99810);d=4;embedding=torch.randn(13,d)*magnitude;u=torch.linalg.qr(torch.randn(d,2)).Q;b=embedding@u;i,j=torch.triu_indices(2,2);cache=dict(u=u,projected=b,pair_indices=(i,j),pair_products=b[:,i]*b[:,j]);squares=embedding.square();function=UpdateFunction(5*d+2,width=4)
        with torch.no_grad():function.net[-1].weight.normal_(0,.05)
        named=list(function.named_parameters());sizes=[v.numel() for _,v in named];flat=torch.cat([v.detach().flatten() for _,v in named]).requires_grad_();bases=torch.randn(4,13);hidden=torch.randn(4,d);targets=[2,7,3,9];initial=torch.cat([torch.zeros(2*d),torch.ones(1)])
        def step(state,parameters,t):
            chunks=parameters.split(sizes);values={name:chunk.reshape(value.shape) for (name,value),chunk in zip(named,chunks)}
            return future_step(lambda x:functional_call(function,values,(x,)),state,embedding,squares,bases[t],hidden[t],targets[t],t,cache)
        def rollout(parameters):
            state=initial;losses=[]
            for t in range(4):
                loss,state,_=step(state,parameters,t);losses.append(loss)
            return torch.stack(losses).mean()
        exact=torch.autograd.grad(rollout(flat),flat)[0];state=initial;eligibility=torch.zeros(len(initial),len(flat));accumulated=torch.zeros_like(flat)
        for t in range(4):
            state=state.detach().requires_grad_();local_loss,next_state,_=step(state,flat,t);cot=torch.autograd.grad(local_loss,state)[0];accumulated+=cot@eligibility/4
            jac_state,jac_function=torch.autograd.functional.jacobian(lambda s,p:step(s,p,t)[1],(state,flat));eligibility=jac_state@eligibility+jac_function;state=next_state.detach()
        error=float((exact-accumulated).abs().max());assert error<1e-11
        finite=[]
        for index in range(len(flat)-7,len(flat)):
            plus=flat.detach().clone();minus=plus.clone();plus[index]+=1e-6;minus[index]-=1e-6;finite.append(abs(float((rollout(plus)-rollout(minus))/2e-6-exact[index])))
        assert max(finite)<1e-7;checks.append(dict(embedding_scale=magnitude,eligibility_error=error,finite_difference_error=max(finite)))
    paths=['experiments/credit/check_multistep_geometric_update.py','experiments/credit/multistep_geometric_update.py','experiments/credit/online_geometric_update.py']
    result=dict(status='completed',battle='R1',tag=a.tag,gate=True,checks=checks,wall_s=time.monotonic()-start,source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},scope='Four-target complete lexical/momentum/product state derivative, feedback/feature/geometry dependence included; forward eligibility matches reverse graph and finite differences. Synthetic13class4dimensional head, no native depth/task quality claim.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
