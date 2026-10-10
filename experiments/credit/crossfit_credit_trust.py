"""B1/R1 TRAIN-only cross-fitted utility trust, then fresh native assessment."""
import argparse,hashlib,json,resource,time
import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode
from fit_optimizer_aware_credit import Correction,collect,ROOT,native
from shared_prefix_receiver_credit import SharedPrefixCredit
from gated_learning_effect import bound_packet,select
SOURCE='experiments/results/credit/curie_optimizer_aware_fit_20261011T0010Z_pilot.json'
SELECTION='experiments/results/credit/curie_credit_trust_20261011T0130Z_select.json'

def fit(data,steps):
    target=data['aware']-data['fixed'];scale=float(target.square().mean().sqrt().clamp_min(1e-8));torch.manual_seed(170);head=Correction(scale).double();opt=torch.optim.AdamW(head.parameters(),lr=.003,weight_decay=.001);actions=native.base.actions()[None,None].expand(len(target),6,-1,-1)
    for step in range(steps):
        pred=head.values(data['z'],actions);loss=((pred-target)/scale).square().mean();assert torch.isfinite(loss);opt.zero_grad();loss.backward();opt.step()
    return head,opt

def selection(saved,smoke):
    data=saved['train_tensors'];count=8 if smoke else 32;folds=2 if smoke else 4;steps=2 if smoke else 256;data={k:v[:count] for k,v in data.items()};order=np.random.default_rng(191).permutation(count);splits=np.array_split(order,folds);alphas=[0.,.25,.5,1.];gains=np.zeros((count,4));fold_records=[]
    for held in splits:
        train=np.setdiff1d(np.arange(count),held);head,_=fit({k:v[train] for k,v in data.items()},steps)
        with torch.no_grad():
            z=data['z'][held];base=data['base'][held];truth=data['aware'][held].flatten(1);action=native.base.actions()[None,None].expand(len(held),6,-1,-1);effect=head.values(z,action);chosen=base.flatten(1).argmin(1);before=truth.gather(1,chosen[:,None]).squeeze(1)
            for j,alpha in enumerate(alphas):
                chosen=(base+alpha*effect).flatten(1).argmin(1);after=truth.gather(1,chosen[:,None]).squeeze(1);gains[held,j]=(before-after).numpy()
        fold_records.append(dict(fit_indices=train.tolist(),held_indices=held.tolist()))
    bootstrap=np.random.default_rng(192).integers(0,count,(4000,count));means=gains.mean(0);low=np.quantile(gains[bootstrap].mean(1),.05/3,axis=0);valid=[j for j in range(1,4) if low[j]>0];chosen=max(valid,key=lambda j:means[j]) if valid else 0;head,opt=fit(data,steps)
    return dict(chosen_alpha=alphas[chosen],alphas=alphas,crossfit_mean_gains=means.tolist(),exploratory_one_sided_lower=low.tolist(),folds=fold_records,steps=steps,train_prefixes=count,selection_rule='Max mean utility among positive-alpha candidates with positive multiplicity-adjusted exploratory lower bootstrap; else0. Fold models overlap, so not a coverage guarantee.'),head,opt

def fresh(saved):
    result=json.loads((ROOT/SELECTION).read_text());assert result['status']=='completed' and result['mode']=='select';cp=ROOT/result['checkpoint'];assert hashlib.sha256(cp.read_bytes()).hexdigest()==result['checkpoint_sha256'];chosen=torch.load(cp,map_location='cpu',weights_only=False);head=Correction(chosen['scale']).double();head.load_state_dict(chosen['head']);alpha=result['metrics']['chosen_alpha']
    source=json.loads((ROOT/SOURCE).read_text());old=json.loads((ROOT/source['prior_result']).read_text());baseline=torch.load(ROOT/old['checkpoint'],map_location='cpu',weights_only=False);model=native.actor(4,170);model.load_state_dict(baseline['actor']);critic=SharedPrefixCredit(model.M,native.prior.bound(model)).double();critic.load_state_dict(baseline['critic']);critic.requires_grad_(False)
    z,base,static,truth=collect(model,critic,470,32,8,list(range(8)),.05,True);actions=native.base.actions()[None,None].expand(32,8,-1,-1);rows=[];values={};indices=np.random.default_rng(193).integers(0,32,(4000,32));fixed=int(saved['train_tensors']['aware'].mean(0).flatten().argmin());flat=truth.flatten(1)
    with torch.no_grad():
        with FlopCounterMode(display=False) as preparation:packet=bound_packet(head,.25,0)
        for name,a in [('preserved_baseline',0.),('crossfit_trust',alpha),('ungated_paired',1.)]:
            with FlopCounterMode(display=False) as query:choice,need=select(base,z,actions,head,a,packet,0)
            v=flat.gather(1,choice[:,None]).squeeze(1);values[name]=v;gain=(flat[:,fixed]-v).numpy();rows.append(dict(arm=name,alpha=a,correction_prefixes=int(need.sum()),correction_supported_flops=query.get_total_flops(),gain_vs_train_fixed=float(gain.mean()),gain_bootstrap95=np.quantile(gain[indices].mean(1),[.025,.975]).tolist(),gain_vs_noop=float(-v.mean()),chosen_receiver_action=torch.stack((choice//9,choice%9),1).tolist()))
        gain=(values['preserved_baseline']-values['crossfit_trust']).numpy();untrusted=(values['preserved_baseline']-values['ungated_paired']).numpy()
    return dict(chosen_alpha=alpha,rows=rows,gain_trusted_over_baseline=float(gain.mean()),gain_bootstrap95=np.quantile(gain[indices].mean(1),[.025,.975]).tolist(),gain_ungated_over_baseline=float(untrusted.mean()),ungated_bootstrap95=np.quantile(untrusted[indices].mean(1),[.025,.975]).tolist(),bound_preparation_supported_flops=preparation.get_total_flops(),correction_bound=packet.value,fresh_prefixes=32,future_rollouts=8,stream_seed=470,actor_seed=170,selection_checkpoint_sha256=result['checkpoint_sha256']),dict(z=z,base=base,static=static,aware=truth)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);ap.add_argument('--mode',choices=['smoke','select','fresh'],required=True);a=ap.parse_args();torch.set_num_threads(1);torch.set_default_dtype(torch.float64);out=ROOT/'experiments/results/credit'/f'{a.tag}.json';assert not out.exists();begin=time.monotonic();source=json.loads((ROOT/SOURCE).read_text());cp=ROOT/source['checkpoint'];assert hashlib.sha256(cp.read_bytes()).hexdigest()==source['checkpoint_sha256'];saved=torch.load(cp,map_location='cpu',weights_only=False)
    with FlopCounterMode(display=False) as counter:
        if a.mode=='fresh':metrics,packets=fresh(saved);payload=dict(packets=packets)
        else:metrics,head,opt=selection(saved,a.mode=='smoke');payload=dict(head=head.state_dict(),optimizer=opt.state_dict(),scale=head.scale)
    checkpoint=out.with_suffix('.pt');torch.save(payload,checkpoint);sources=['experiments/credit/crossfit_credit_trust.py','experiments/credit/gated_learning_effect.py','experiments/credit/fit_optimizer_aware_credit.py','experiments/credit/optimizer_aware_route_credit.py','experiments/credit/shared_prefix_receiver_credit.py',*native.SOURCES]
    result=dict(status='completed',battle='B1/R1',tag=a.tag,mode=a.mode,metrics=metrics,whole_wall_s=time.monotonic()-begin,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,supported_flops=counter.get_total_flops(),checkpoint=str(checkpoint.relative_to(ROOT)),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),source_checkpoint_sha256=source['checkpoint_sha256'],source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},scope='One actor seed170, synthetic readout-only learning worlds. Trust uses TRAIN cross-fitting only; fresh stream seed470 generated after locked selection. No receiver6/7 TRAIN labels. Guard bootstrap is exploratory with overlapping folds. Fresh baseline descriptors/teacher replay and all earlier fitting remain charged; residual query work separated. No online critic co-adaptation, full actor learning, matched-quality efficiency, hardware or TEST claim.');out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(metrics,indent=2))
if __name__=='__main__':main()
