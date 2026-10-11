"""R1 finite learned-feature/optimizer credit identities before meta-learner fitting."""
import argparse,hashlib,json,time
from pathlib import Path
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[2]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args();torch.set_num_threads(1);torch.set_default_dtype(torch.float64);start=time.monotonic();torch.manual_seed(1007)
    E=torch.randn(5,3)*.4;z=torch.tensor([.05,-.07,.03]);base=torch.tensor([.2,-.6,.1,.8,-.3]);r=2;beta=.4;target=3;eta=.05
    loss=lambda state,offset:F.cross_entropy((offset+E@state)[None],torch.tensor([target]))
    probabilities=(base+E@z).softmax(0);mu=probabilities@E;g=mu-E[target]
    covariance=E.T@(probabilities[:,None]*E)-mu[:,None]*mu[None]
    hessian=torch.autograd.functional.hessian(lambda state:loss(state,base),z)
    errors=dict(hessian_covariance=float((hessian-covariance).abs().max()))
    changed=base+F.one_hot(torch.tensor(r),5)*beta;other=(changed+E@z).softmax(0);othermu=other@E
    excluded=(mu-probabilities[r]*E[r])/(1-probabilities[r])
    shift=(other[r]-probabilities[r])*(E[r]-excluded)
    errors['route_mean_shift']=float((shift-(othermu-mu)).abs().max())
    B=torch.randn(3,3)*.2;P=(B@B.T+.2*torch.eye(3)).requires_grad_();next_z=z-eta*P@g.detach()
    future=torch.tensor([-.2,.1,.3,-.5,.8]);future_target=4
    objective=F.cross_entropy((future+E@next_z)[None],torch.tensor([future_target]))
    actual=torch.autograd.grad(objective,P)[0]
    future_mu=(future+E@next_z.detach()).softmax(0)@E;lam=future_mu-E[future_target]
    expected=-eta*lam[:,None]*g[None]
    errors['future_loss_optimizer_derivative']=float((actual-expected).abs().max())
    Q=torch.tensor([[2.,.1,0.],[0.,1.,.2],[.1,0.,.7]]);inverse=Q.inverse();EE=E@Q;zz=inverse@z;PP=inverse@P.detach()@inverse.T
    gp=(base+EE@zz).softmax(0)@EE-EE[target];next_transformed=zz-eta*PP@gp
    errors['coordinate_prediction']=float((EE@zz-E@z).abs().max())
    errors['coordinate_update']=float((EE@next_transformed-E@next_z.detach()).abs().max())
    errors['coordinate_kernel']=float((EE@PP@EE.T-E@P.detach()@E.T).abs().max())
    eigen,rotation=torch.linalg.eigh(P.detach());sqrt=(rotation*eigen.sqrt()[None])@rotation.T;invsqrt=(rotation*eigen.rsqrt()[None])@rotation.T
    jacobian=torch.eye(3)-eta*P.detach()@covariance;weighted=invsqrt@jacobian@sqrt
    norm=float(torch.linalg.svdvals(weighted).max());R=float(E.norm(dim=1).max());assert eta*float(eigen.max())*R**2<=1 and norm<=1+1e-12
    next_b=z-eta*P.detach()@g;next_a=z-eta*P.detach()@(othermu-E[target]);delta=next_a-next_b
    future_loss=lambda state:F.cross_entropy((future+E@state)[None],torch.tensor([future_target]))
    lambda_b=(future+E@next_b).softmax(0)@E-E[future_target]
    exact_difference=float(future_loss(next_a)-future_loss(next_b));linear=float(lambda_b@delta)
    kernel=float(-eta*(other[r]-probabilities[r])*(lambda_b@P.detach()@(E[r]-excluded)))
    errors['learning_mediated_kernel']=abs(linear-kernel);remainder=abs(exact_difference-linear);bound=.5*R**2*float(delta.square().sum());assert remainder<=bound+1e-14
    assert max(errors.values())<1e-12,errors
    result=dict(status='completed',battle='R1',tag=a.tag,gate=True,errors=errors,weighted_update_norm=norm,future_loss_remainder=remainder,feature_bound=bound,whole_wall_s=time.monotonic()-start,source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['experiments/credit/check_learned_credit_geometry.py','experiments/theory/LEARNED_FEATURE_GEOMETRY_OF_CREDIT.md']},scope='Finite fixed-feature head: Hessian/covariance, intervention mean shift, issued-snapshot next-loss derivative for preconditioner, nonorthogonal coordinate/kernel/update invariance, fixed-metric nonexpansion and learning-mediated Taylor remainder. No Adam stability/general semantic-grokking/optimizer fitting or native language-quality claim; dense-feedback meta-learning contracts only.')
    out=ROOT/f'experiments/results/credit/{a.tag}.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
