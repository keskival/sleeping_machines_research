"""R1/B1 crossed label-noise/history diagnostic, synthetic generator only.
Frozen native actor; no actor fitting, route proposals, TEST or scaling claim.
"""
import argparse,hashlib,json,resource,time
import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode
import future_route_critic_v1 as base
from conditional_credit_noise import analytic_advantages,continuation
SOURCES=['experiments/credit/averaged_future_credit_v3.py','experiments/credit/conditional_credit_noise.py',*base.SOURCES]

def collect(model,seed,count,rollouts,dev=False):
    packets=[];first=[];average=[];index=3
    with torch.no_grad():
        for i in range(count):
            t,m=base.batch((700000 if dev else 600000)+seed*1000+i,size=1,pairs=2)
            packet=base.prefix_packet(model,t,m,index);packets.append(packet)
            rng=np.random.default_rng((9000000 if dev else 8000000)+seed*10000+i)
            labels=[]
            for k in range(rollouts):
                ft,fm=continuation(t,m,2,rng)
                # Input packets use only the original prefix, never its suffix.
                labels.append(analytic_advantages(model,ft,fm,index,base.actions())[0])
            first.append(labels[0]);average.append(torch.stack(labels).mean(0))
    return torch.cat(packets),torch.stack(first),torch.stack(average)

def run(a):
    model=base.actor(2,a.seed)
    x,one,y=collect(model,a.seed,a.train_packets,a.train_rollouts)
    dx,done,dy=collect(model,a.seed,a.dev_packets,a.dev_rollouts,True)
    scale=float(y.square().mean().sqrt().clamp_min(1e-8));actions=base.actions()[None].expand(len(x),-1,-1)
    fixed=y.mean(0);fixed_one=one.mean(0);selected=int(fixed.argmin());reference_error=(dy-fixed).square().mean(-1)
    rng=np.random.default_rng(a.seed);indices=rng.integers(0,len(dy),(2000,len(dy)))
    rows=[]
    for connected in (False,True):
        for label,target in [('single',one),('averaged',y)]:
            torch.manual_seed(a.seed);critic=base.Critic(connected).double();critic.output_scale.fill_(scale)
            opt=torch.optim.AdamW(critic.parameters(),lr=.003,weight_decay=.001)
            for k in range(a.steps):
                pred=critic.values(x,actions);loss=((pred-target)/scale).square().mean()
                opt.zero_grad();loss.backward();opt.step();assert torch.isfinite(loss)
            with torch.no_grad():
                pred=critic.values(dx,base.actions()[None].expand(len(dx),-1,-1));chosen=pred.argmin(-1)
                values=dy.gather(1,chosen[:,None]).squeeze(1);errors=(pred-dy).square().mean(-1)
                mse_gain=(reference_error-errors).numpy();utility_gain=(dy[:,selected]-values).numpy()
                rows.append(dict(history='connected' if connected else 'isolated',labels=label,parameters=sum(p.numel() for p in critic.parameters()),train_relative_mse=float((critic.values(x,actions)-target).square().sum()/target.square().sum().clamp_min(1e-30)),dev_relative_mse=float((pred-dy).square().sum()/dy.square().sum().clamp_min(1e-30)),dev_future_advantage=float(values.mean()),mse_gain_vs_averaged_train_constant=float(mse_gain.mean()),utility_gain_vs_averaged_train_constant=float(utility_gain.mean()),mse_gain_bootstrap95=np.quantile(mse_gain[indices].mean(1),[.025,.975]).tolist(),utility_gain_bootstrap95=np.quantile(utility_gain[indices].mean(1),[.025,.975]).tolist(),chosen_actions=chosen.tolist(),chosen_advantages=values.tolist(),dev_predictions=pred.tolist()))
    return dict(arms=rows,train_mean_reference=fixed.tolist(),single_train_mean_reference=fixed_one.tolist(),reference_selected_action=selected,reference_dev_relative_mse=float((dy-fixed).square().sum()/dy.square().sum().clamp_min(1e-30)),reference_dev_future_advantage=float(dy[:,selected].mean()),dev_branch_mean_advantages=dy.tolist(),shared_train_scale=scale,teacher_actor_forwards=a.train_packets*(a.train_rollouts+1)+a.dev_packets*(a.dev_rollouts+1),teacher_target_presentations=a.train_packets*(7*a.train_rollouts+3)+a.dev_packets*(7*a.dev_rollouts+3),scope='Crossed history/label averaging, identical initial weights and common TRAIN-derived scale. All four arms share teacher generation. DEV evaluates finite independent continuation averages; bootstrap is prefix uncertainty conditional on fitted seed. Averaged TRAIN constant is the stronger common comparator; single TRAIN constant saved. Synthetic true-generator access is diagnostic, not a deployment capability. No actor weight updates or asynchronous learner.',gate='All three seeds must improve DEV calibration and intervention selection against averaged TRAIN constant before depth scaling; no automated scaling.')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);ap.add_argument('--mode',choices=['smoke','pilot'],required=True);ap.add_argument('--seed',type=int,default=170);ap.add_argument('--steps',type=int,default=256);ap.add_argument('--train-packets',type=int,default=128);ap.add_argument('--dev-packets',type=int,default=64);ap.add_argument('--train-rollouts',type=int,default=8);ap.add_argument('--dev-rollouts',type=int,default=16);a=ap.parse_args()
    if a.mode=='smoke':a.steps=2;a.train_packets=a.dev_packets=2;a.train_rollouts=a.dev_rollouts=2
    out=base.ROOT/'experiments/results/credit'/f'{a.tag}.json';assert not out.exists();torch.set_num_threads(1);torch.set_default_dtype(torch.float64);start=time.monotonic()
    with FlopCounterMode(display=False) as counter:metrics=run(a)
    result=dict(status='completed',tag=a.tag,battle='R1/B1',args=vars(a),metrics=metrics,wall_s=time.monotonic()-start,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,supported_flops=counter.get_total_flops(),flop_scope='All shared teacher generation and four critic fits; supported PyTorch operations only, special/unsupported operations excluded. No work-to-quality claim.',source_sha256={p:hashlib.sha256((base.ROOT/p).read_bytes()).hexdigest() for p in SOURCES})
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='metrics'}),flush=True)
if __name__=='__main__':main()
