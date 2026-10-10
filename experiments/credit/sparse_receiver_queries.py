"""B1/R1 saved-critic native query-budget measurement; no fitting."""
import argparse,hashlib,json,resource,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode
import unseen_receiver_credit as native
ROOT=native.ROOT
TAGS={170:'curie_unseen_receiver_v2_20261010T2140Z_pilot',171:'curie_unseen_receiver_confirm_s171_20261010T2125Z',172:'curie_unseen_receiver_confirm_s172_20261010T2125Z'}

def receivers(anchor,budget,seed):
    # Partial Fisher-Yates over implicit non-anchor indices: O(budget) draws,
    # without scoring/reading descriptions of unqueried receivers.
    rng=np.random.default_rng(seed);swaps={};chosen=[anchor]
    for pos in range(budget-1):
        j=int(rng.integers(pos,7));v=swaps.get(j,j);swaps[j]=swaps.get(pos,pos);chosen.append(v if v<anchor else v+1)
    assert len(set(chosen))==budget
    return chosen

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);ap.add_argument('--mode',choices=['smoke','measure'],required=True);a=ap.parse_args();out=ROOT/'experiments/results/credit'/f'{a.tag}.json';assert not out.exists();torch.set_num_threads(1);torch.set_default_dtype(torch.float64);start=time.monotonic();rows=[];bindings=[];total_teacher_flops=0;seeds=[170] if a.mode=='smoke' else [170,171,172];count=2 if a.mode=='smoke' else 32;rollouts=2 if a.mode=='smoke' else 8
    with torch.no_grad():
        for seed in seeds:
            source=ROOT/'experiments/results/credit'/f'{TAGS[seed]}.json';old=json.loads(source.read_text());cp=ROOT/old['metrics']['checkpoint'];assert hashlib.sha256(cp.read_bytes()).hexdigest()==old['metrics']['checkpoint_sha256']
            for path,digest in old['source_sha256'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
            saved=torch.load(cp,map_location='cpu',weights_only=False);model=native.actor(saved['args']['depth'],seed);model.load_state_dict(saved['actor']);anchor=old['metrics']['fixed_train_receiver'];fixed=old['metrics']['fixed_train_action'];bindings.append(dict(seed=seed,result=str(source.relative_to(ROOT)),result_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),checkpoint=str(cp.relative_to(ROOT)),checkpoint_sha256=old['metrics']['checkpoint_sha256'],prior_whole_fit_flops=old['supported_flops'],prior_whole_fit_wall_s=old['wall_s']))
            critics={}
            structured=native.prior.KernelCritic(model.M,saved['query_bound']).double();structured.encoder=nn.GRU(saved['feature_width'],32,batch_first=True).double();critics['structured']=structured
            free=native.prior.FreeCritic(saved['free_width'],saved['train_scale']).double();free.encoder=nn.GRU(saved['feature_width'],32,batch_first=True).double();critics['free']=free
            for name,critic in critics.items():critic.load_state_dict(saved['models'][name]);critic.eval()
            data=[];teacher_start=time.monotonic()
            with FlopCounterMode(display=False) as teacher:
                for i in range(count):
                    t,m=native.base.batch(3000000+seed*1000+i,size=1,pairs=2);rng=np.random.default_rng(33000000+seed*1000+i);ys=[];first=None
                    for j in range(rollouts):
                        ft,fm=native.prior.continuation(t,m,2,rng);v=native.responses(model,ft,fm,3,list(range(8)),native.base.actions())[0];ys.append(v)
                        if first is None:first=(ft,fm,v)
                    samples=torch.stack(ys);data.append((t,m,samples.mean(0),samples.var(0,unbiased=True),first))
            teacher_wall=time.monotonic()-teacher_start;total_teacher_flops+=teacher.get_total_flops()
            bootstrap=np.random.default_rng(seed).integers(0,count,(2000,count))
            for name,critic in critics.items():
                results={}
                for budget in [1,2,4,8]:
                    gains=[];errors=[];noise=[];selected=[];parity=0.;policy_wall=0.;flops=0;replay_flops=0
                    for i,(t,m,truth,var,first) in enumerate(data):
                        tick=time.monotonic()
                        with FlopCounterMode(display=False) as policy:
                            routes=receivers(anchor,budget,34000000+seed*1000+i);x=native.prefix(model,t,m,3,routes)[0];pred=critic.values(x,native.base.actions()[None].expand(budget,-1,-1));pred=pred.clone();pred[:,0]=0.;best=int(pred.flatten().argmin());r=routes[best//9];action=best%9
                        policy_wall+=time.monotonic()-tick;flops+=policy.get_total_flops();chosen=float(truth[r,action]);gains.append(float(truth[anchor,fixed])-chosen);selected.append([r,action]);errors.append(float((pred-truth[routes]).square().mean()));noise.append(float(var[routes].mean()))
                        if i<2:
                            ft,fm,first_truth=first
                            with FlopCounterMode(display=False) as replay:
                                before,_=native.losses(model,ft,fm);changed,_=native.losses(model,ft,fm,3,torch.tensor([r]),native.base.actions()[action:action+1]);parity=max(parity,abs(float((changed[:,3:]-before[:,3:]).mean()-first_truth[r,action])));assert torch.equal(before[:,:3],changed[:,:3])
                            replay_flops+=replay.get_total_flops()
                    assert parity<1e-10;g=np.array(gains);results[budget]=g
                    rows.append(dict(seed=seed,critic=name,budget=budget,prefixes=count,queried_receiver_packets=count*budget,proposal_draws=count*(budget-1),gain_vs_train_fixed=float(g.mean()),gain_bootstrap95=np.quantile(g[bootstrap].mean(1),[.025,.975]).tolist(),gain_vs_noop=float(np.mean([float(d[2][anchor,fixed]) for d in data])-g.mean())*-1,prediction_mse_against_rollout_mean=float(np.mean(errors)),within_prefix_response_variance=float(np.mean(noise)),conditional_mean_error_unbiased_estimate=float(np.mean(errors)-np.mean(noise)/rollouts),policy_wall_s=policy_wall,policy_supported_flops=flops,selected_write_replay_supported_flops=replay_flops,native_write_replay_error=parity,chosen_routes_actions=selected,teacher_wall_s=teacher_wall,teacher_supported_flops=teacher.get_total_flops()))
                dense=results[8]
                for row in [r for r in rows if r['seed']==seed and r['critic']==name]:
                    diff=dense-results[row['budget']];row['gain_difference_dense_minus_budget']=float(diff.mean());row['difference_bootstrap95']=np.quantile(diff[bootstrap].mean(1),[.025,.975]).tolist()
    sources=['experiments/credit/sparse_receiver_queries.py',*native.SOURCES]
    result=dict(status='completed',battle='B1/R1',tag=a.tag,args=vars(a),rows=rows,saved_fit_bindings=bindings,whole_wall_s=time.monotonic()-start,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,teacher_supported_flops=total_teacher_flops,source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},scope='Saved nearly matched critics; no refit. Fresh synthetic prefixes/continuations, frozen depth4 native actor. TRAIN-best receiver anchor plus uniform implicit-index candidates, not learned acquisition. Candidate budget includes anchor; optional no-op score exactly zero for both critics. Policy work includes native prefix recomputation, which retains dense eight-mark forward scoring, all queried descriptors and critic work; no dense critic-score filtering. Teacher/replay costs and previous fitting are separate and charged. Timing includes Python FlopCounter instrumentation and is not production throughput or asynchronous hardware. Unbiased conditional-mean error estimates can be negative from finite continuations. No full actor learning, real-stream TEST or whole-fit efficiency claim.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps([ {k:r[k] for k in ['seed','critic','budget','gain_vs_train_fixed','policy_supported_flops']} for r in rows],indent=2))
if __name__=='__main__':main()
