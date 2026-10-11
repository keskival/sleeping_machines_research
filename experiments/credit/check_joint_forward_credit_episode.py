"""B1/R1 numerical gate before coupled forward-derived route learning."""
import argparse,hashlib,json,time
from pathlib import Path
import torch
from joint_forward_credit_episode import episode
from reference_learner_route_credit import shadow_scores
from shared_carrier_paired_replay import replay_pairs
from sparse_paired_actor_critic import initial,native
ROOT=Path(__file__).resolve().parents[2]
SOURCES=['experiments/credit/joint_forward_credit_episode.py','experiments/credit/check_joint_forward_credit_episode.py','experiments/credit/shared_carrier_paired_replay.py','experiments/credit/reference_learner_route_credit.py',*native.SOURCES]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args()
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64);start=time.monotonic();checks=[]
    carrier,*_=initial()
    for depth in [2,4,8]:
        model=carrier if depth==4 else native.actor(depth,170);model.requires_grad_(False)
        t,m=native.base.batch(995000+depth,size=1,pairs=2);routes=[1,7];actions=native.base.actions()[[3,8]]
        joint=episode(model,t,m,routes,actions,0,1,.05,True,True)
        actual,weights,detail=replay_pairs(model,t,m,routes,actions,.05,True)
        shadow,_,sd=shadow_scores(model,t,m,routes,actions,1,.05,True)
        errors=dict(actual=float((joint['actual_scores']-actual).abs().max()),shadow=float((joint['projected_scores']-shadow).abs().max()),static=float((joint['frozen_scores']-sd['frozen_scores']).abs().max()))
        for k,row in enumerate([1,0]):
            errors[f'weights{k}']=max(float((weights[n][row]-v).abs().max()) for n,v in joint['head_states'][k].items())
            errors[f'gradients{k}']=max(float((d[n][row]-joint['gradients'][j][k][n]).abs().max()) for j,d in enumerate(detail['gradients']) for n in d)
            errors[f'moments{k}']=max(float((detail['optimizer']['state'][n][field][row]-joint['optimizers'][k]['state'][n][field]).abs().max()) for n in joint['optimizers'][k]['state'] for field in ['exp_avg','exp_avg_sq'])
        changed=m.clone();changed[:,5:]=(changed[:,5:]+1)%8
        later=episode(model,t,changed,routes,actions,0,1,.05)
        errors['causality']=float((later['projected_scores'][:,0]-joint['projected_scores'][:,0]).abs().max())
        identity=episode(model,t,m,routes,torch.zeros_like(actions),0,1,.05)
        errors['identity']=float((identity['actual_scores'][0]-identity['actual_scores'][1]).abs().max())
        assert identity['actual_worlds']==1 and identity['readout_updates']==4
        assert max(errors.values())<1e-11;checks.append(dict(depth=depth,errors=errors))
    out=ROOT/f'experiments/results/credit/{a.tag}.json';assert not out.exists()
    result=dict(status='completed',battle='B1/R1',tag=a.tag,contracts=checks,gate=True,whole_wall_s=time.monotonic()-start,source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCES},scope='One carrier, two independent actual readout learners; reference-only detached all-route projections. Actual scores/gradients/Adam/weights, frozen and reference projections, suffix causality, physical-identity reuse contracted at depth2/4/8. No fit or TEST.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
