"""Transport no-op/zero-step/first-update and exact reference Adam contracts."""
import argparse,hashlib,json,time
import torch
from check_token_forward_credit import ROOT,KeyedTokenLM
from token_forward_credit import packet
from token_persistent_credit import joint_replay_carry
from token_exact_world_readouts import replay
from token_optimizer_transport import replay_transport


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args();torch.set_num_threads(1);torch.set_default_dtype(torch.float64);start=time.monotonic()
    out=ROOT/f'experiments/results/credit/{a.tag}.json';assert not out.exists();checks=[]
    for depth in [2,4,8]:
        for reset in [False,True]:
            torch.manual_seed(100100+depth);m=KeyedTokenLM(8,3,depth,2,4,1,0.);m.embed=torch.nn.Embedding(23,8);m.bias=torch.nn.Parameter(torch.randn(23)*.02);m.requires_grad_(False);m.eval()
            tokens=torch.tensor([2,4,6,8,2,4,6,8,2,4,6,8]);
            if reset:tokens[7]=7
            p=packet(m,tokens,index=3,eos_token=7);actions=torch.zeros(3,2,4);actions[1,0,1]=.25;actions[2,1,2]=-.25;receivers=[2,4,22]
            reference=joint_replay_carry(p,receivers,actions,0,0,.01,diagnostics=True)
            frozen=replay(p,receivers,actions,0.)
            for curvature in ['zero','diagonal']:
                t=replay_transport(p,receivers,actions,.01,curvature,True)
                errors=dict(reference_scores=float((t['reference_scores']-reference['actual_scores'][0]).abs().max()),reference_z=float((t['reference_z']-reference['final_z'][0]).abs().max()),noop_z=float((t['z'][0]-t['reference_z']).abs().max()),noop_scores=float((t['scores'][0]-t['reference_scores']).abs().max()))
                exact=reference['optimizers'][0]['state'][0];transport=t['reference_optimizer']['state'][0]
                errors['reference_adam']=max(float((exact[k]-transport[k]).abs().max()) for k in ['step','exp_avg','exp_avg_sq'])
                zero=replay_transport(p,receivers,actions,0.,curvature)
                errors['zero_lr']=float((zero['scores']-frozen['scores']).abs().max())
                first=type(p)(p.log_probabilities[:1],p.queries[:1],p.targets[:1],p.reset[:1],p.effect_live[:1],p.embedding,p.rates,p.dk,p.index,p.state,False)
                actual=replay(first,receivers,actions,.01);one=replay_transport(first,receivers,actions,.01,curvature)
                errors['first_branch_update']=float((actual['z']-one['z']).abs().max())
                assert max(errors.values())<1e-10,errors
                checks.append(dict(depth=depth,reset=reset,curvature=curvature,errors=errors))
    paths=['experiments/credit/check_token_optimizer_transport.py','experiments/credit/token_optimizer_transport.py','experiments/credit/token_issued_credit_geometry.py']
    result=dict(status='completed',battle='R1',tag=a.tag,gate=True,checks=checks,whole_wall_s=time.monotonic()-start,source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},scope='Small native depths2/4/8, EOS, zero learning, no-op, first true branch Adam update and full exact reference learner scores/state/moments. Longer alternative states/losses approximate; these identities do not assert transport accuracy or utility.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
