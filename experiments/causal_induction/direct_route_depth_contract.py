"""B5/R1 conditional score-error and actual-Adam utility-response contracts."""
import copy
import json
import resource
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from native_sensor import ROOT
from world_credit_pipeline import RESULTS,sha
from world_credit_confirmation_verify import verify_result
from sleeping_machines.paired_route_credit import paired_route_credit
TAG='curie_direct_route_depth_20261012_v1_contract'

def main():
    torch.set_num_threads(1);begin=time.monotonic();out=RESULTS/(TAG+'.json');assert not out.exists()
    parent=verify_result('curie_world_credit_confirm_20261012_v1_s194_decision');assert parent['confirmation_gate']
    generator=torch.Generator().manual_seed(88301)
    score_error=producer_error=pair_error=0.;bound_margin=pair_margin=float('inf');cases=0
    for count in [2,4,8,32]:
        for repeat in range(8):
            scores=torch.randn(count,generator=generator,dtype=torch.float64,requires_grad=True)
            z=torch.randn(count,generator=generator,dtype=torch.float64)*.7
            pi=scores.softmax(0);mu=.3;epsilon=.07;q=mu+epsilon
            actual=(pi*(F.softplus(z)-mu*z)).sum();predicted=(pi*(F.softplus(z)-q*z)).sum()
            truth=torch.autograd.grad(actual,scores,retain_graph=True)[0]
            approximate=torch.autograd.grad(predicted,scores,retain_graph=True)[0]
            bias=approximate-truth;mean=(pi*z).sum();formula=-epsilon*pi*(z-mean)
            score_error=max(score_error,float((bias-formula).abs().max().detach()))
            spread=float((z.max()-z.min()).detach());bound_margin=min(bound_margin,abs(epsilon)*spread/2-float(bias.abs().sum()))
            jacobian=torch.randn(count,11,generator=generator,dtype=torch.float64)
            jbar=(pi.detach()[:,None]*jacobian).sum(0)
            theta_bias=jacobian.T@bias.detach()
            covariance=-epsilon*(pi.detach()[:,None]*(z-mean.detach())[:,None]*(jacobian-jbar)).sum(0)
            producer_error=max(producer_error,float((theta_bias-covariance).abs().max()))
            variance=(pi.detach()*(z-mean.detach()).square()).sum()
            jvariance=(pi.detach()*(jacobian-jbar).square().sum(1)).sum()
            assert float(theta_bias.norm())<=abs(epsilon)*float((variance*jvariance).sqrt())+1e-12
            J=torch.diag(pi.detach())-pi.detach()[:,None]*pi.detach()[None,:]
            epsilon_pair=(torch.rand(count,count,generator=generator,dtype=torch.float64)-.5)*.08
            pair_mean=torch.zeros(count,dtype=torch.float64)
            for i in range(count):
                for j in range(count):
                    if i==j:continue
                    delta=F.softplus(z[j])-F.softplus(z[i])-(mu+epsilon_pair[i,j])*(z[j]-z[i])
                    pair_mean+=pi[i].detach()*J[j]*delta
                    if count==4 and repeat==0:
                        surrogate=paired_route_credit(pi[None],torch.tensor([j]),z.new_tensor([1/(count-1)]),delta[None])
                        observed=torch.autograd.grad(surrogate,scores,retain_graph=True)[0]
                        assert float((observed-J[j]*delta*(count-1)).abs().max())<1e-12
            pair_bias=pair_mean-truth.detach()
            pair_formula=torch.zeros_like(pair_bias)
            for i in range(count):
                for j in range(count):
                    if i!=j:pair_formula-=pi[i].detach()*J[j]*epsilon_pair[i,j]*(z[j]-z[i])
            pair_error=max(pair_error,float((pair_bias-pair_formula).abs().max()))
            pair_margin=min(pair_margin,.08/2*2*spread-float(pair_bias.abs().sum()))
            cases+=1
    assert score_error<1e-12 and producer_error<1e-12 and pair_error<1e-12
    assert bound_margin>=-1e-12 and pair_margin>=-1e-12
    adam_parity=utility_error=finite_difference_error=0.;active=[]
    for scale in [.03,3.]:
        parameter=torch.nn.Parameter(torch.randn(7,generator=generator,dtype=torch.float64))
        optimizer=torch.optim.Adam([parameter],lr=.003,betas=(.9,.99),eps=1e-8)
        for _ in range(3):
            parameter.grad=torch.randn(7,generator=generator,dtype=torch.float64)*.1;optimizer.step();optimizer.zero_grad()
        theta=parameter.detach().clone();state=copy.deepcopy(optimizer.state[parameter]);step=int(state['step'])+1
        g0=torch.randn(7,generator=generator,dtype=torch.float64)*scale
        directions=torch.randn(7,3,generator=generator,dtype=torch.float64)*.1
        means=torch.tensor([.2,.4,.6],dtype=torch.float64,requires_grad=True);issued=means.detach().clone();batch=4;cap=.2
        def update(m):
            raw=g0-directions@(m-issued)/batch
            factor=(cap/(raw.norm()+1e-6)).clamp(max=1.)
            clipped=raw*factor
            moment=.9*state['exp_avg']+.1*clipped
            second=.99*state['exp_avg_sq']+.01*clipped.square()
            return theta-.003*(moment/(1-.9**step))/((second/(1-.99**step)).sqrt()+1e-8)
        proposed=update(means)
        parameter.grad=g0.clone();torch.nn.utils.clip_grad_norm_([parameter],cap);optimizer.step()
        adam_parity=max(adam_parity,float((proposed.detach()-parameter.detach()).abs().max()))
        active.append(bool(float(g0.norm())+1e-6>cap))
        A=torch.randn(7,7,generator=generator,dtype=torch.float64);H=A.T@A;offset=torch.randn(7,generator=generator,dtype=torch.float64)
        terminal=.5*proposed@H@proposed+offset@proposed
        direct=torch.autograd.grad(terminal,means,retain_graph=True)[0]
        future=(H@proposed+offset).detach()
        raw=g0.detach().clone().requires_grad_();factor=(cap/(raw.norm()+1e-6)).clamp(max=1.);clipped=raw*factor
        response=theta-.003*((.9*state['exp_avg']+.1*clipped)/(1-.9**step))/(((.99*state['exp_avg_sq']+.01*clipped.square())/(1-.99**step)).sqrt()+1e-8)
        cotangent=torch.autograd.grad(response,raw,grad_outputs=future)[0]
        formula=-directions.T@cotangent/batch
        utility_error=max(utility_error,float((direct-formula).abs().max()))
        h=1e-5
        def utility(m):
            value=update(m);return float((.5*value@H@value+offset@value).detach())
        for k in range(3):
            plus=issued.clone();minus=issued.clone();plus[k]+=h;minus[k]-=h
            finite_difference_error=max(finite_difference_error,abs((utility(plus)-utility(minus))/(2*h)-float(direct[k])))
    assert active==[False,True] and adam_parity<1e-12 and utility_error<1e-12 and finite_difference_error<1e-7
    sources=['experiments/causal_induction/direct_route_depth_contract.py','experiments/theory/DIRECT_ROUTE_CREDIT_DEPTH_AND_DELAYED_UTILITY.md','sleeping_machines/paired_route_credit.py','experiments/causal_induction/world_credit_confirmation_verify.py']
    result=dict(status='completed',battle='B5/R1',tag=TAG,checks=dict(score_cases=cases,score_bias_identity_error=score_error,producer_covariance_identity_error=producer_error,
        fixed_belief_bound_min_margin=bound_margin,pair_conditioned_bias_identity_error=pair_error,pair_bound_min_margin=pair_margin,
        actual_clipped_Adam_proposal_error=adam_parity,delayed_utility_identity_error=utility_error,utility_finite_difference_error=finite_difference_error,clipping_cases=active),
        source_sha256={n:sha(ROOT/n) for n in sources},parent_sha256={str((RESULTS/(parent['tag']+'.json')).relative_to(ROOT)):sha(RESULTS/(parent['tag']+'.json'))},
        torch_version=torch.__version__,whole_pipeline_wall_s=time.monotonic()-begin,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope='Small float64 algebra/executable operator contracts, scores2/4/8/32 and clipped/unclipped actual Adam. Supports conditional score bounds and delayed scalar teaching, not calibrated world accuracy, sparse hardware cost, a new trained optimizer or neural-depth scaling. Original native confirmation unchanged; dummy optimizer steps are numerical fixtures, no benchmark fitting.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['checks']),flush=True)
if __name__=='__main__':main()
