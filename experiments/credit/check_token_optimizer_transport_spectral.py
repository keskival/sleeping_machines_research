"""Carried shared-gradient transport versus exact native reference Adam."""
import argparse,hashlib,json,time
from copy import deepcopy
import torch
from torch.nn import functional as F
from check_token_forward_credit import ROOT,KeyedTokenLM
from token_forward_credit import packet,bonuses
from token_persistent_credit import joint_replay_carry
from token_optimizer_transport_spectral import replay_transport
from token_spectral_curvature import basis,covariance_action,projected_variances
from token_issued_credit_geometry import route_mean_shifts

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args();torch.set_num_threads(1);torch.set_default_dtype(torch.float64);start=time.monotonic()
    out=ROOT/f'experiments/results/credit/{a.tag}.json';assert not out.exists();checks=[]
    for depth in [2,4,8]:
        for reset in [False,True]:
            torch.manual_seed(103100+depth);m=KeyedTokenLM(8,3,depth,2,4,1,0.);m.embed=torch.nn.Embedding(23,8);m.bias=torch.nn.Parameter(torch.randn(23)*.02);m.requires_grad_(False);m.eval()
            tokens=torch.tensor([2,4,6,8,2,4,6,8,2,4,6,8,2,4,6,8,2,4,6,8])
            if reset:tokens[14]=7
            actions=torch.zeros(3,2,4);actions[1,0,1]=.25;actions[2,1,2]=-.25;receivers=[2,4,22]
            first=packet(m,tokens[:12],index=3,eos_token=7)
            acquired=joint_replay_carry(first,receivers,actions,1,0,.01)
            incoming=acquired['factual_learner'];before=deepcopy(incoming)
            p=packet(m,tokens[11:],index=0,initial_state=acquired['factual_carrier'],eos_token=7)
            ref=joint_replay_carry(p,receivers,actions,0,0,.01,initial_learner=incoming,diagnostics=True)
            for curvature in ['zero','diagonal','spectral']:
                t=replay_transport(p,receivers,actions,.01,curvature,True,incoming)
                errors=dict(scores=float((t['reference_scores']-ref['actual_scores'][0]).abs().max()),reference_state=float((t['reference_z']-ref['final_z'][0]).abs().max()),noop_state=float((t['z'][0]-t['reference_z']).abs().max()))
                exact=ref['optimizers'][0]['state'][0];transport=t['reference_optimizer']['state'][0]
                errors['moments']=max(float((exact[k]-transport[k]).abs().max()) for k in ['step','exp_avg','exp_avg_sq'])
                one=type(p)(p.log_probabilities[:1],p.queries[:1],p.targets[:1],p.reset[:1],p.effect_live[:1],p.embedding,p.rates,p.dk,p.index,p.state,False)
                approximation=replay_transport(one,receivers,actions,.01,curvature,False,incoming)
                beta=bonuses(one,actions)
                truth=[]
                for i,r in enumerate(receivers):
                    z=torch.nn.Parameter(incoming['z'].clone());opt=torch.optim.Adam([z],lr=.01);opt.load_state_dict(deepcopy(incoming['optimizer']))
                    logits=one.log_probabilities[0]+one.embedding@z+F.one_hot(torch.tensor(r),23)*beta[i,0]
                    F.cross_entropy(logits[None],one.targets).backward();opt.step();truth.append(z.detach())
                errors['first_alternative_update']=float((approximation['z']-torch.stack(truth)).abs().max())
                assert max(errors.values())<1e-10,errors
                assert torch.equal(incoming['z'],before['z'])
                for k,v in before['optimizer']['state'].items():
                    for field,value in v.items():assert torch.equal(value,incoming['optimizer']['state'][k][field])
                checks.append(dict(depth=depth,reset=reset,curvature=curvature,errors=errors))
    torch.manual_seed(104101);embedding=torch.randn(23,8);probability=torch.full((23,),1/23);mean=probability@embedding
    exact=embedding.T@(probability[:,None]*embedding)-mean[:,None]*mean[None]
    cache=basis(embedding,8);variance=probability@cache['projected_square']-(probability@cache['projected']).square()
    diagonal=exact.diag();ident=torch.eye(8)
    matrix=covariance_action(diagonal[None].expand(8,-1),variance[None].expand(8,-1),cache['u'],ident)
    covariance_error=float((matrix-exact).abs().max());assert covariance_error<1e-10
    eigmin=float(torch.linalg.eigvalsh(matrix).min());assert eigmin>=-1e-10
    checks.append(dict(uniform_fullrank_covariance_error=covariance_error,minimum_eigenvalue=eigmin))
    logits=torch.randn(23);receivers=[0,1,2];beta=torch.tensor([0.,.7,-.4]);shift,dp,_=route_mean_shifts(logits,embedding,receivers,beta)
    var=projected_variances(logits.softmax(0),receivers,dp,cache)
    for i,r in enumerate(receivers):
        tilted=(logits+F.one_hot(torch.tensor(r),23)*beta[i]).softmax(0);b=cache['projected'];truevar=tilted@b.square()-(tilted@b).square()
        assert float((var[i]-truevar).abs().max())<1e-10
        mu=tilted@embedding;h=embedding.T@(tilted[:,None]*embedding)-mu[:,None]*mu[None]
        matrix=covariance_action(h.diag()[None].expand(8,-1),var[i:i+1].expand(8,-1),cache['u'],ident)
        assert float(torch.linalg.eigvalsh((matrix+matrix.T)/2).min())>=-1e-10
        assert float((matrix.diag()-h.diag()).abs().max())<1e-10
    paths=['experiments/credit/token_optimizer_transport_spectral.py','experiments/credit/token_spectral_curvature.py','experiments/credit/check_token_optimizer_transport_spectral.py']
    result=dict(status='completed',battle='R1',tag=a.tag,gate=True,checks=checks,whole_wall_s=time.monotonic()-start,source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},scope='After8 real fast-head Adam updates and selected hard write, carried reference/no-op moments/state/scores and first independent alternative update agree at small native depths2/4/8, with causal EOS reset. Incoming snapshots unchanged. Longer alternative trajectories remain approximations; no task/depth-quality claim.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
