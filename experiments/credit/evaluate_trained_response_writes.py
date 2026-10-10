"""R1/B1 saved future-credit derivatives guide fresh native key-write proposals.
No critic refitting; bounded proposals are explicitly uncertified for real futures.
"""
import argparse,hashlib,json,resource,time
import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode
import structured_future_credit_v5 as prior
from response_credit_policy import activations,propose
PREFIX='curie_structured_credit_v5b_20261010T1845Z'
SOURCES=['experiments/credit/evaluate_trained_response_writes.py','experiments/credit/response_credit_policy.py',*prior.SOURCES]

def run(a):
    record_path=prior.base.ROOT/'experiments/results/credit'/f'{PREFIX}_pilot_s{a.seed}.json';record=json.loads(record_path.read_text());path=prior.base.ROOT/record['metrics']['checkpoint'];assert hashlib.sha256(path.read_bytes()).hexdigest()==record['metrics']['checkpoint_sha256']
    saved=torch.load(path,map_location='cpu',weights_only=False);model=prior.base.actor(4,a.seed);model.load_state_dict(saved['actor_state']);structured=prior.KernelCritic(model.M,saved['query_bound']).double();free=prior.FreeCritic(saved['free_width'],saved['train_scale']).double()
    for name,critic in [('structured',structured),('free',free)]:
        critic.load_state_dict(saved['models'][name]);critic.eval()
        for p in critic.parameters():p.requires_grad_(False)
    gen=torch.Generator().manual_seed(181);extra=torch.randn(16,4,generator=gen);extra=.25*extra/extra.norm(dim=-1,keepdim=True);pool=torch.cat((prior.base.actions(),extra))
    # Reconstruct the unchanged TRAIN-mean action; charge all reference generation.
    _,train,_=prior.collect(model,a.seed,saved['args']['train_packets'],saved['args']['train_rollouts'],pool)
    fixed=int(train.mean(0).argmin());values=[];parity=0.;activation_size=0
    for i in range(a.prefixes):
        t,m=prior.base.batch(1700000+a.seed*1000+i,size=1,pairs=2)
        with torch.no_grad():
            x=prior.base.prefix_packet(model,t,m,3);packet=activations(structured,x,record['metrics']['checkpoint_sha256'],'structured_saved');delta,info=propose(packet,packet['actor_version'],packet['credit_version']);assert info['status']=='bounded_uncertified'
            activation_size=sum(v.numel() for v in packet.values() if isinstance(v,torch.Tensor))
        z=torch.zeros(1,1,4,requires_grad=True);g=torch.autograd.grad(free.values(x,z).sum(),z)[0].reshape(1,4).detach();L=saved['query_bound']**2/4;eta=torch.minimum(torch.full_like(g.norm(dim=-1),1/L),.25/g.norm(dim=-1).clamp_min(1e-30));free_delta=-eta[:,None]*g
        assert delta.norm()<=.25+1e-12 and free_delta.norm()<=.25+1e-12
        candidates=torch.cat((pool,free_delta,delta));rng=np.random.default_rng(18000000+a.seed*10000+i);out=[]
        with torch.no_grad():
            for k in range(a.rollouts):
                ft,fm=prior.continuation(t,m,2,rng);response=prior.decode(*prior.response(model,ft,fm,3),candidates[None]).mean(1)[0];out.append(response)
                if i<4 and k==0:
                    original,_=prior.base.event_losses(model,ft,fm)
                    for j,action in enumerate((free_delta,delta)):
                        changed,_=prior.base.event_losses(model,ft,fm,3,action);parity=max(parity,abs(float((changed[:,3:]-original[:,3:]).mean()-response[25+j])));assert torch.equal(original[:,:3],changed[:,:3])
        values.append(torch.stack(out).mean(0))
    y=torch.stack(values);reference=y[:,fixed];rng=np.random.default_rng(a.seed);indices=rng.integers(0,len(y),(2000,len(y)));arms=[]
    for name,index in [('free_gradient',25),('structured_cotangent',26)]:
        gain=(reference-y[:,index]).numpy();noop=-y[:,index].numpy();arms.append(dict(arm=name,mean_future_advantage=float(y[:,index].mean()),gain_vs_fixed_train_action=float(gain.mean()),gain_vs_fixed_bootstrap95=np.quantile(gain[indices].mean(1),[.025,.975]).tolist(),gain_vs_noop=float(noop.mean()),per_prefix_advantages=y[:,index].tolist()))
    paired=(y[:,25]-y[:,26]).numpy();N=saved['args']['train_packets'];M=saved['args']['train_rollouts'];audit_calls=min(4,a.prefixes)*3
    return dict(arms=arms,structured_gain_vs_free=float(paired.mean()),structured_gain_vs_free_bootstrap95=np.quantile(paired[indices].mean(1),[.025,.975]).tolist(),native_selected_write_parity_error=parity,retained_response_activation_elements=activation_size,proposal_dimension=4,train_fixed_action=fixed,reference_generation_actor_calls=N*(M+1),fresh_evaluation_actor_calls=a.prefixes*(a.rollouts+1)+audit_calls,teacher_target_presentations=N*(7*M+3)+a.prefixes*(7*a.rollouts+3)+audit_calls*7,checkpoint_sha256=record['metrics']['checkpoint_sha256'],prior_training_supported_flops=record['supported_flops'],scope='Saved critics and frozen native actor; fresh independent prefixes/continuations, single bounded proposal per critic. TRAIN reference regenerated and charged; all 25 reference directions plus two proposals evaluated through exact local responses. Four actual writes per critic checked by native replay. Prefix bootstrap conditional on fitted seed, not seed uncertainty. Proposals have no certified population-gradient error bound. Forward weights and credit weights do not update; this measures state-write utility, not online actor learning, async scheduling or hardware savings.')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);ap.add_argument('--seed',type=int,default=170);ap.add_argument('--mode',choices=['smoke','measure'],required=True);ap.add_argument('--prefixes',type=int,default=64);ap.add_argument('--rollouts',type=int,default=16);a=ap.parse_args()
    if a.mode=='smoke':a.prefixes=4;a.rollouts=2
    output=prior.base.ROOT/'experiments/results/credit'/f'{a.tag}.json';assert not output.exists();torch.set_num_threads(1);torch.set_default_dtype(torch.float64);start=time.monotonic()
    with FlopCounterMode(display=False) as counter:metrics=run(a)
    result=dict(status='completed',tag=a.tag,battle='R1/B1',args=vars(a),metrics=metrics,wall_s=time.monotonic()-start,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,supported_flops=counter.get_total_flops(),flop_scope='All reference regeneration, cotangent/proposal inference, fresh response evaluation and selected native writes; excludes special/unsupported arithmetic. Prior fitting work reported separately; no amortized savings claim.',source_sha256={p:hashlib.sha256((prior.base.ROOT/p).read_bytes()).hexdigest() for p in SOURCES});output.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
