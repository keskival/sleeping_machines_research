"""B1/R1 information-path comparison on paid TRAIN packets and fresh DEV futures."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import resource
import time
import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode
from sparse_paired_actor_critic import initial, native, scores as prefix_scores, probabilities
from cached_paired_native_replay import replay_pairs
from revealed_future_route_credit import RevealedFutureCredit,observed_future,scores,contrasts

ROOT=Path(__file__).resolve().parents[2]
SOURCE='experiments/results/credit/curie_sparse_paired_ac_v3_20261011T0727Z_reuse.json'
TAG='curie_future_packet_20261011T0739Z'
SOURCES=['experiments/credit/fit_revealed_future_credit.py','experiments/credit/revealed_future_route_credit.py',
         'experiments/credit/cached_paired_native_replay.py','experiments/credit/shared_prefix_receiver_credit.py',
         'experiments/credit/sparse_paired_actor_critic.py',*native.SOURCES]


def fresh_model(carrier,baseline):
    torch.manual_seed(361)
    model=RevealedFutureCredit(carrier.M,native.prior.bound(carrier)).double()
    loaded=model.load_state_dict(baseline.state_dict(),strict=False)
    assert not loaded.unexpected_keys and all(k.startswith(('future.','future_projection.')) for k in loaded.missing_keys)
    return model


def contract(carrier,baseline):
    t,m=native.base.batch(931337,size=1,pairs=2)
    ft,fm=native.prior.continuation(t,m,2,np.random.default_rng(931338))
    with torch.no_grad():x=native.prefix(carrier,t,m,3,list(range(8)))
    u=observed_future(ft,fm);model=fresh_model(carrier,baseline)
    full=scores(model,x,u,0);old=prefix_scores(baseline,x,0)
    initial_error=float((full-old).abs().max());assert initial_error<1e-12
    selected=contrasts(model,x,u,[13],[65],0)
    sparse_error=float((selected-(full.flatten(1)[:,13]-full.flatten(1)[:,65])).abs().max())
    g1=torch.autograd.grad(selected.sum(),tuple(model.parameters()))
    g2=torch.autograd.grad((full.flatten(1)[:,13]-full.flatten(1)[:,65]).sum(),tuple(model.parameters()))
    gradient_error=max(float((a-b).abs().max()) for a,b in zip(g1,g2));assert max(sparse_error,gradient_error)<1e-11
    packet=model.encode_packet(x[:,0,:,:30],u,0,2);rejects=0
    try:model.values(packet,x[:,:,:,30:],native.base.actions()[None,None].expand(1,8,-1,-1),0,3)
    except ValueError:rejects+=1
    try:model.encode_packet(x[:,0,:,:30],u[:,:3],0,2)
    except ValueError:rejects+=1
    assert rejects==2
    with torch.no_grad():
        model.future_projection.weight.normal_(0,.05)
        dependency=float((scores(model,x,u,0)-scores(model,x,torch.zeros_like(u),0)).abs().max())
    assert dependency>1e-12
    return dict(initial_prior_value_error=initial_error,sparse_contrast_error=sparse_error,sparse_gradient_error=gradient_error,
                stale_or_partial_packet_rejections=rejects,nonzero_future_dependence=dependency,
                connected_critic_parameters=sum(p.numel() for p in model.parameters()),prefix_prior_parameters=sum(p.numel() for p in baseline.parameters()))


def run(mode,carrier,baseline,tau,scale,source,checkpoint):
    began=time.monotonic();xs=[];us=[];acts=[];refs=[];ys=[];reconstruction_errors=[]
    data=[h for h in checkpoint['states']['sparse_learned']['history'] if h['extra_world']]
    if mode=='smoke':data=data[:8]
    with torch.no_grad(),FlopCounterMode(display=False) as preparation:
        for n,h in enumerate(data):
            i=h['prefix'];t,m=native.base.batch(144701000+i,size=1,pairs=2)
            ft,fm=native.prior.continuation(t,m,2,np.random.default_rng(75000000+i))
            xs.append(native.prefix(carrier,t,m,3,list(range(6))));us.append(observed_future(ft,fm))
            acts.append(h['action']);refs.append(h['reference']);ys.append(h['observed_contrast'])
            if n<3:
                loss,_=replay_pairs(carrier,ft,fm,[h['action']//9,h['reference']//9],native.base.actions()[[h['action']%9,h['reference']%9]],.05)
                error=abs(float((loss[0]-loss[1]).mean())-h['observed_contrast']);assert error<1e-11
                reconstruction_errors.append(error)
    x=torch.cat(xs);u=torch.cat(us);y=torch.tensor(ys);split=int(.75*len(data));assert split>=4 and split<len(data)
    prep=dict(packets=len(data),train_packets=split,chronological_holdout_packets=len(data)-split,
              supported_flops=preparation.get_total_flops(),wall_s=time.monotonic()-began,
              first_three_reconstructed_label_errors=reconstruction_errors)
    initial_model=fresh_model(carrier,baseline)
    models={name:deepcopy(initial_model) for name in ['revealed','zero_future']};work=[];opts={}
    steps=4 if mode=='smoke' else 256
    for name,model in models.items():
        opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001);tick=time.monotonic()
        with FlopCounterMode(display=False) as fitting:
            for step in range(steps):
                ix=np.random.default_rng(983000+step).choice(split,min(16,split),replace=False)
                future=u[ix] if name=='revealed' else torch.zeros_like(u[ix])
                prediction=contrasts(model,x[ix],future,[acts[i] for i in ix],[refs[i] for i in ix],step)
                loss=((prediction-y[ix])/scale).square().mean();assert torch.isfinite(loss)
                opt.zero_grad();loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(norm);opt.step()
        work.append(dict(arm=name,parameters=sum(p.numel() for p in model.parameters()),updates=steps,
                         supported_flops=fitting.get_total_flops(),wall_s=time.monotonic()-tick))
        opts[name]=opt.state_dict()
    holdout=[]
    with torch.no_grad(),FlopCounterMode(display=False) as validation:
        prior=(prefix_scores(baseline,x,0).flatten(1)[torch.arange(len(x)),torch.tensor(acts)]-
               prefix_scores(baseline,x,0).flatten(1)[torch.arange(len(x)),torch.tensor(refs)])
        predictions={'prefix_prior':prior}
        for name,model in models.items():predictions[name]=contrasts(model,x,u if name=='revealed' else torch.zeros_like(u),acts,refs,steps)
        for name,pred in predictions.items():
            holdout.append(dict(arm=name,train_mse=float((pred[:split]-y[:split]).square().mean()),
                chronological_holdout_mse=float((pred[split:]-y[split:]).square().mean())))
    actor=deepcopy(baseline);actor.load_state_dict(checkpoint['states']['sparse_fixed']['actor']);actor.requires_grad_(False)
    count=2 if mode=='smoke' else 32;futures=1 if mode=='smoke' else 8
    truth=[];predicted={name:[] for name in ['prefix_prior',*models]};unseen=[];weights=[];teacher_work=0;tick=time.monotonic()
    with torch.no_grad(),FlopCounterMode(display=False) as evaluation:
        for i in range(count):
            t,m=native.base.batch(146823000+i,size=1,pairs=2);px=native.prefix(carrier,t,m,3,list(range(8)))
            base=prefix_scores(baseline,px,0).flatten();reference=int(base.argmin())
            probe=int(np.random.default_rng(83000000+i).integers(72))
            if probe==reference or probe%9==reference%9==0:probe=(probe+1)%72
            pi=probabilities(prefix_scores(actor,px,512),tau).flatten()
            soft=(pi-.1/72)/.9;one=torch.nn.functional.one_hot(torch.tensor(probe),72).to(pi.dtype)
            score=(-.9/tau)*(soft[probe]/pi[probe])*(one-soft)
            weights.append(float(72*pi[probe]*score.square().sum()))
            unseen.append(probe//9>=6 or reference//9>=6);rng=np.random.default_rng(84000000+i)
            actual=[];guesses={name:[] for name in predicted}
            for _ in range(futures):
                ft,fm=native.prior.continuation(t,m,2,rng);uf=observed_future(ft,fm)
                guesses['prefix_prior'].append(float(base[probe]-base[reference]))
                for name,model in models.items():
                    all_values=scores(model,px,uf if name=='revealed' else torch.zeros_like(uf),steps).flatten()
                    guesses[name].append(float(all_values[probe]-all_values[reference]))
                with FlopCounterMode(display=False) as teacher:
                    losses,_=replay_pairs(carrier,ft,fm,[probe//9,reference//9],native.base.actions()[[probe%9,reference%9]],.05)
                teacher_work+=teacher.get_total_flops();actual.append(float((losses[0]-losses[1]).mean()))
            truth.append(actual)
            for name in predicted:predicted[name].append(guesses[name])
    target=np.array(truth);rows=[];bootstrap=np.random.default_rng(853).integers(0,count,(4000,count))
    for name,pred in predicted.items():
        error=(np.array(pred)-target)**2
        rows.append(dict(arm=name,mse=float(error.mean()),error_over_zero=float(error.sum()/np.sum(target**2)),
            unseen_probe_mse=float(error[np.array(unseen)].mean()) if any(unseen) else None,
            policy_logit_residual_second_moment=float((error*np.array(weights)[:,None]).mean())))
    improvement=((np.array(predicted['zero_future'])-target)**2-(np.array(predicted['revealed'])-target)**2).mean(1)
    metric=dict(preparation=prep,fit_work=work,chronological_holdout=holdout,dev_rows=rows,
        revealed_over_zero_future_mse_gain=float(improvement.mean()),prefix_bootstrap95=np.quantile(improvement[bootstrap].mean(1),[.025,.975]).tolist(),
        dev_prefixes=count,futures_per_prefix=futures,paid_teacher_worlds=2*count*futures,
        evaluation_wall_s=time.monotonic()-tick,evaluation_supported_flops=evaluation.get_total_flops(),
        evaluation_teacher_supported_flops=teacher_work,validation_supported_flops=validation.get_total_flops(),
        steps=steps,critic_scale=scale,tau=tau,
        inherited_source_fit_costs=source['metrics']['cumulative_fits'],inherited_source_whole_wall_s=source['whole_wall_s'])
    payload=dict(models={k:v.state_dict() for k,v in models.items()},optimizers=opts,train_packets=dict(x=x,future=u,actions=acts,references=refs,y=y,split=split),
                 dev_truth=truth,dev_predictions=predicted,dev_logit_importance_weights=weights)
    return metric,payload


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);ap.add_argument('--mode',choices=['contract','smoke','measure'],required=True);a=ap.parse_args()
    torch.set_num_threads(1);torch.set_default_dtype(torch.float64);out=ROOT/f'experiments/results/credit/{a.tag}.json';assert not out.exists();start=time.monotonic()
    source=json.loads((ROOT/SOURCE).read_text());assert source['status']=='completed'
    cp=ROOT/source['checkpoint'];assert hashlib.sha256(cp.read_bytes()).hexdigest()==source['checkpoint_sha256']
    checkpoint=torch.load(cp,weights_only=False,map_location='cpu')
    assert all(len(s['history'])==s['cursor']==512 for s in checkpoint['states'].values())
    carrier,baseline,tau,scale,_=initial()
    if a.mode=='measure':
        for mode in ['contract','smoke']:
            d=json.loads((ROOT/f'experiments/results/credit/{TAG}_{mode}.json').read_text());assert d['status']=='completed'
            for p,digest in d['source_sha256'].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==digest,p
    if a.mode=='contract':metrics,payload=contract(carrier,baseline),None
    else:metrics,payload=run(a.mode,carrier,baseline,tau,scale,source,checkpoint)
    result=dict(status='completed',battle='B1/R1',tag=a.tag,mode=a.mode,metrics=metrics,
        whole_wall_s=time.monotonic()-start,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        source_result=SOURCE,source_result_sha256=hashlib.sha256((ROOT/SOURCE).read_bytes()).hexdigest(),source_checkpoint_sha256=source['checkpoint_sha256'],
        source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCES},
        scope='Connected learning-side future encoder and whole-response critic, capacity-matched zero-future control, same initialization/paid TRAIN labels/fit updates. Last25% paid TRAIN packets chronologically held out; fresh32-prefix/eight-future DEV contrasts use per-future labels. Raw observed marks/gaps are exogenous to internal writes; actor never receives future inputs and is not fitted here. All72 critic values queried per DEV future to charge a supported model-gradient term; random uniform paired probes measure residuals. Policy-logit second moment uses saved fixed sparse512 controller and importance weights, not complete native-parameter gradient variance. Fixed temporal/keyed depth4 carrier and four readout Adam updates; full fitting, reconstruction and teacher work separately recorded, inherited source acquisition additional. Supported FLOPs exclude unsupported/scalar/optimizer arithmetic; wall includes it. No actor utility, benchmark, TEST, all-carrier, changing-regime or depth-scaling claim.')
    if payload is not None:
        saved=out.with_suffix('.pt');torch.save(payload,saved);result.update(checkpoint=str(saved.relative_to(ROOT)),checkpoint_sha256=hashlib.sha256(saved.read_bytes()).hexdigest())
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
