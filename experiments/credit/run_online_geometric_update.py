"""R1 native streaming predictions with online forward-trained update functions."""
import argparse,hashlib,json,resource,time
from copy import deepcopy
import numpy as np
import torch
from check_token_forward_credit import ROOT,SOURCE,KeyedTokenLM,load
from measure_token_forward_credit import verify
from token_episode_state_vectorized import consume
from token_block_curvature import block_basis,block_covariances,block_action
from online_geometric_update import UpdateFunction,OnlineUpdateLearner,geometric_step
from token_flop_accounting import TokenFlopCounter

def new_function(d):
    torch.manual_seed(99710);return UpdateFunction(5*d+2)

def curvature(probability,embedding,squares,cache):
    mean=probability@embedding;diagonal=(probability@squares-mean.square()).clamp_min(0)
    covariance=block_covariances(probability,[0],probability.new_zeros(1),cache)
    d=embedding.shape[1];identity=torch.eye(d,dtype=embedding.dtype)
    matrix=block_action(diagonal[None].expand(d,-1),covariance.expand(d,-1,-1),cache['u'],identity)
    return (matrix+matrix.T)/2

def episode(model,tokens,cache,initial=None,pretrain=False):
    d=model.embed.embedding_dim;embedding=model.embed.weight.detach();squares=embedding.square();hidden={}
    hook=model.out_norm.register_forward_hook(lambda module,args,result:hidden.update(value=result.detach().flatten()))
    names=['online'] if pretrain else ['adam','fixed','frozen','online'];learners={}
    for name in names:
        if name=='fixed':
            learners[name]=dict(z=embedding.new_zeros(d),moment=embedding.new_zeros(d),product=embedding.new_tensor(1.));continue
        if name=='adam':
            z=torch.nn.Parameter(embedding.new_zeros(d));learners[name]=dict(z=z,opt=torch.optim.Adam([z],lr=.01))
        else:
            function=new_function(d)
            if initial is not None and name in ['frozen','online']:function.load_state_dict(initial['function'])
            state=None if initial is None or name!='online' else deepcopy(initial['optimizer'])
            learners[name]=OnlineUpdateLearner(function,d,name=='online',state)
    state=None;rows={name:[] for name in names};stats={name:[] for name in names};common_time=0.;arm_times={name:0. for name in names};key_counts=[];common_flops=0;arm_flops={name:0 for name in names};begin=time.monotonic()
    try:
        with TokenFlopCounter(display=False) as work:
            for j in range(len(tokens)-1):
                if int(tokens[j])==50256:
                    state=None
                    for name,learner in learners.items():
                        if name=='adam':
                            z=torch.nn.Parameter(embedding.new_zeros(d));learners[name]=dict(z=z,opt=torch.optim.Adam([z],lr=.01))
                        elif name=='fixed':learners[name]=dict(z=embedding.new_zeros(d),moment=embedding.new_zeros(d),product=embedding.new_tensor(1.))
                        else:learner.reset()
                tick=time.monotonic();before=work.get_total_flops()
                with torch.no_grad():forecast,state=consume(model,tokens[j:j+1],state)
                common_time+=time.monotonic()-tick;common_flops+=work.get_total_flops()-before
                if j<63:continue
                target=int(tokens[j+1]);key_counts.append(len(state['keys']))
                for name in names:
                    tick=time.monotonic();before=work.get_total_flops();learner=learners[name];z=learner['z'].detach() if name in ['adam','fixed'] else learner.z
                    with torch.no_grad():
                        logits=forecast['log_probabilities'][0]+embedding@z;probability=logits.softmax(0);nll=float(-logits.log_softmax(0)[target]);g=probability@embedding-embedding[target]
                    assert np.isfinite(nll);rows[name].append(nll)
                    if name=='adam':
                        learner['opt'].zero_grad();learner['z'].grad=g.clone();learner['opt'].step()
                    else:
                        h=curvature(probability,embedding,squares,cache)
                        if name=='fixed':
                            with torch.no_grad():nz,nm,nr=geometric_step(z,learner['moment'],learner['product'],g,h,z.new_tensor(.01),z.new_tensor(1.),z.new_tensor(.9))
                            learners[name]=dict(z=nz,moment=nm,product=nr)
                        else:stats[name].append(learner.feedback(g,h,hidden['value'],nll,j-63))
                    arm_times[name]+=time.monotonic()-tick;arm_flops[name]+=work.get_total_flops()-before
            flops=work.get_total_flops()
    finally:hook.remove()
    online=learners['online'];checkpoint=dict(function=deepcopy(online.function.state_dict()),optimizer=deepcopy(online.optimizer.state_dict()))
    summary=dict(wall_s=time.monotonic()-begin,supported_flops=flops,common_native_wall_s=common_time,common_native_supported_flops=common_flops,arm_wall_s=arm_times,arm_supported_flops=arm_flops,scored_targets=len(tokens)-64,meta_updates=online.meta_updates,mean_nll={name:float(np.mean(values)) for name,values in rows.items()},max_scored_keys=max(key_counts),mean_scored_keys=float(np.mean(key_counts)),selected_native_address_writes=len(tokens)-1,lexical_capacity=d,native_available_vocabulary=len(embedding),update_function_parameters=sum(v.numel() for v in online.function.parameters()))
    return checkpoint,summary,dict(losses=rows,hyperparameters=stats)

def bootstrap(values):
    v=np.array(values);rng=np.random.default_rng(99711);b=v[rng.integers(0,len(v),(10000,len(v)))].mean(1)
    return dict(gain=float(v.mean()),paired_episode_bootstrap95=np.quantile(b,[.025,.975]).tolist(),per_episode=v.tolist())

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);ap.add_argument('--mode',choices=['smoke','fit'],required=True);a=ap.parse_args();start=time.monotonic();torch.set_num_threads(1);torch.set_default_dtype(torch.float64)
    out=ROOT/f'experiments/results/credit/{a.tag}.json';assert not out.exists();assert verify('curie_online_geometric_update_20261011_v1_contract')['gate']
    if a.mode=='fit':assert verify('curie_online_geometric_update_20261011_v1_smoke')['gate']
    source=json.loads((ROOT/SOURCE).read_text());c=source['args'];native=ROOT/source['checkpoint'];m=KeyedTokenLM(c['d'],c['modes'],c['layers'],c['heads'],c['dk'],c['keyed'],c['dropout']);m.load_state_dict(torch.load(native,weights_only=True));m.requires_grad_(False);m.eval()
    tick=time.monotonic()
    with TokenFlopCounter(display=False) as work:cache=block_basis(m.embed.weight.detach(),8)
    basis=dict(wall_s=time.monotonic()-tick,supported_flops=work.get_total_flops(),cached_bytes=sum(v.numel()*v.element_size() for v in cache.values()))
    data=load(ROOT/'data/fineweb_gpt2/fineweb_train_000001.bin');train_targets,episodes,targets=(64,2,32) if a.mode=='smoke' else (1024,4,512)
    offset=c['train_tokens']+720896;tokens=torch.tensor(np.array(data[offset:offset+64+train_targets],dtype=np.int64))
    initial,pretraining,train_record=episode(m,tokens,cache,pretrain=True);print(json.dumps(dict(stage='pretraining',metrics=pretraining)),flush=True)
    rows=[];records=[];dev_offset=c['train_tokens']+(792576 if a.mode=='smoke' else 786432)
    for i in range(episodes):
        tokens=torch.tensor(np.array(data[dev_offset+1024*i:dev_offset+1024*i+64+targets],dtype=np.int64));_,row,record=episode(m,tokens,cache,initial)
        row['episode']=i;rows.append(row);records.append(record);print(json.dumps(dict(stage='prequential',metrics=row)),flush=True)
    comparisons={f'online_over_{arm}':bootstrap([r['mean_nll'][arm]-r['mean_nll']['online'] for r in rows]) for arm in ['frozen','fixed','adam']}
    cp=out.with_suffix('.pt');torch.save(dict(initial=initial,pretraining_record=train_record,episode_records=records),cp)
    paths=['experiments/credit/run_online_geometric_update.py','experiments/credit/online_geometric_update.py','experiments/credit/ONLINE_GEOMETRIC_UPDATE_PROTOCOL.md','experiments/credit/token_block_curvature.py','experiments/credit/token_spectral_curvature.py','experiments/credit/token_episode_state_vectorized.py']
    result=dict(status='completed',battle='R1',tag=a.tag,mode=a.mode,gate=True,utility_gate=comparisons['online_over_frozen']['paired_episode_bootstrap95'][0]>0,config=dict(train_targets=train_targets,episodes=episodes,targets=targets,context=64,meta_lr=.001,train_offset=offset,private_dev_offset=dev_offset),basis=basis,pretraining=pretraining,rows=rows,comparisons=comparisons,whole_wall_s=time.monotonic()-start,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,checkpoint=str(cp.relative_to(ROOT)),checkpoint_sha256=hashlib.sha256(cp.read_bytes()).hexdigest(),source_checkpoint_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths},scope='ProperGPT2 native2layer keyedsource,64dimensional forwardlexical learner; nativeparameters fixed. Learned connected updatefunction eta/damping/momentum trained by exact one-step futureloss through retained issued graph; frozen-vs-online same inherited function/metaAdamstate. Streaming causal prequential privateTRAINtail; no publicDEV/TEST. No optionalcounterfactual route/actor learning added in this diagnostic; retains source temporal/addressedkey inference. Charge sharednative plus eacharm updates/geometry/function/solver/optimizer, basis/metapretraining additional to everyprimaryarm; unsupported solver/eigensolver/scalar work included in wall but omitted from supportedFLOPs. No equaltrainingclaim vsAdam/fullnativelearning/depth or hardwareefficiencyclaim.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(comparisons=comparisons,utility_gate=result['utility_gate'],whole_wall_s=result['whole_wall_s'])),flush=True)
if __name__=='__main__':main()
