"""G-HIER development: whole-label learning and unseen-combination DEV transfer.

TEST is excluded from the model path. No hidden part supervision. Exact resumable
checkpoints. A short failed gate is about this construction/budget, not the family.
"""
import argparse,gc,json,math,resource,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from experiments.hierarchy.part_combination_task import corpus,audit
from experiments.hierarchy.native_composition_data import sha,source_hashes
from sleeping_machines.frame_native_composition import FrameNativeComposition as CoalescedNativeComposition
from sleeping_machines.prior_centered_readout import fit_prior_centered
from sleeping_machines.convex_native_readout import fit_binary_readout,binary_logits


def forward(model,scene):
    # Only the anonymous sensor interface enters computation.
    events,_=scene.model_input();state=model.new_state()
    for micros,y,x,pol in events:
        model.observe(state,(int(y),int(x)),int(micros)/1e6,[float(pol),1.])
    logits,_=model.compose(state)
    return logits,model.work(state)


class ZeroEmittedContent(CoalescedNativeComposition):
    def _send(self,state,source_level,source_address,value,version,time,kind,target=None):
        if source_level>=0 and kind=='timer':value=value*0
        return super()._send(state,source_level,source_address,value,version,time,kind,target)


def score(model,rows):
    model.eval();saved=torch.get_rng_state();logits=[];work={};features=[]
    hook=model.head.register_forward_pre_hook(lambda module,args:features.append(args[0].detach().clone()))
    with torch.no_grad():
        for row in rows:
            torch.manual_seed(113) # same tape for all inputs, never an example ID
            value,counts=forward(model,row);logits.append(value)
            for key,val in counts.items():
                if isinstance(val,int):work[key]=work.get(key,0)+val
    hook.remove();model.scored_features=torch.stack(features)
    torch.set_rng_state(saved)
    z=torch.stack(logits);y=torch.tensor([r.label for r in rows])
    return z,y,work


def summarize(z,y,calibration):
    logits=calibration['log_prior']+calibration['gain']*(z.double()-calibration['center'])
    losses=F.cross_entropy(logits,y,reduction='none')
    return dict(raw_nll=float(F.cross_entropy(z,y)),calibrated_nll=float(losses.mean()),
        accuracy=float((logits.argmax(-1)==y).double().mean()),losses=losses.tolist())


def atomic_save(path,data):
    temp=path.with_suffix('.writing');torch.save(data,temp);temp.replace(path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--steps',type=int,default=128)
    a=p.parse_args();torch.set_num_threads(1);started=time.monotonic()
    all_rows=corpus();audit(all_rows)
    train=[r for r in all_rows if r.split=='train']
    calibration_rows=[r for r in train if r.nuisance==0]
    dev=[r for r in all_rows if r.split=='dev'] # no TEST model evaluation
    torch.manual_seed(211);model=CoalescedNativeComposition(classes=2)
    old=ROOT/'experiments/results/hierarchy/aws_g_hier_native_composition_20261015_v1_learn.final.pt'
    weights=torch.load(old,map_location='cpu',weights_only=True)
    missing=model.load_state_dict({k:v for k,v in weights.items() if not k.startswith('head.')},strict=False)
    assert set(missing.missing_keys)=={'head.weight','head.bias'} and not missing.unexpected_keys
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    order=torch.Generator().manual_seed(223)
    checkpoint=ROOT/f'experiments/results/hierarchy/{a.tag}.progress.pt'
    work_train={};history=[];start=0;initial=None;past_wall=0.
    initial_cache=ROOT/f'experiments/results/hierarchy/{a.tag}.initial_features.npz'
    if checkpoint.exists():
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert saved['steps']==a.steps
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['rng']);order.set_state(saved['order_rng'])
        assert sha(initial_cache)==saved['initial_features_sha256']
        start=saved['step'];history=saved['history'];initial=saved['initial'];work_train=saved['work_train'];past_wall=saved['wall_s']
    if initial is None:
        z,y,initial_work=score(model,calibration_rows);cal=fit_prior_centered(z,y)
        initial_train_features=model.scored_features.numpy()
        zd,yd,dev_work=score(model,dev)
        np.savez_compressed(initial_cache,train=initial_train_features,dev=model.scored_features.numpy())
        initial=dict(train=summarize(z,y,cal),dev=summarize(zd,yd,cal),gain=cal['gain'],work_train=initial_work,work_dev=dev_work)
    for step in range(start,a.steps):
        model.train();optimizer.zero_grad(set_to_none=True)
        index=int(torch.randint(len(train),(),generator=order));row=train[index]
        logits,counts=forward(model,row);loss=F.cross_entropy(logits[None],torch.tensor([row.label]))
        loss.backward()
        assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        for key,val in counts.items():
            if isinstance(val,int):work_train[key]=work_train.get(key,0)+val
        history.append(float(loss.detach()));del logits,loss
        if (step+1)%8==0 or step+1==a.steps:
            atomic_save(checkpoint,dict(steps=a.steps,step=step+1,model=model.state_dict(),optimizer=optimizer.state_dict(),
                rng=torch.get_rng_state(),order_rng=order.get_state(),initial_features_sha256=sha(initial_cache),history=history,initial=initial,work_train=work_train,
                wall_s=past_wall+time.monotonic()-started))
            print(json.dumps(dict(step=step+1,train_last8=sum(history[-8:])/len(history[-8:]))),flush=True)
    z,y,cal_work=score(model,calibration_rows);cal=fit_prior_centered(z,y)
    final_train_features=model.scored_features.numpy()
    zd,yd,dev_work=score(model,dev)
    final_dev_features=model.scored_features.numpy()
    final=dict(train=summarize(z,y,cal),dev=summarize(zd,yd,cal),gain=cal['gain'])
    intervention=None
    if final['dev']['calibrated_nll']<math.log(2):
        altered=ZeroEmittedContent(classes=2);altered.load_state_dict(model.state_dict())
        za,ya,wa=score(altered,dev);intervention=summarize(za,ya,cal);intervention['work']=wa
        # Resample complete child-code combinations, not correlated nuisance views.
        differences=np.array(intervention['losses'])-np.array(final['dev']['losses'])
        groups={}
        for i,r in enumerate(dev):groups.setdefault((r.left,r.right),[]).append(i)
        means=np.array([differences[idx].mean() for idx in groups.values()]);rng=np.random.default_rng(229)
        boot=means[rng.integers(len(means),size=(2000,len(means)))].mean(1)
        intervention['paired_nll_gain']=float(means.mean());intervention['combination_bootstrap_95']=np.quantile(boot,[.025,.975]).tolist()
    final_weights=ROOT/f'experiments/results/hierarchy/{a.tag}.final.pt';atomic_save(final_weights,model.state_dict())
    feature_path=ROOT/f'experiments/results/hierarchy/{a.tag}_features.npz'
    with np.load(initial_cache) as cached:
        np.savez_compressed(feature_path,initial_train=cached['train'],initial_dev=cached['dev'],
          final_train=final_train_features,final_dev=final_dev_features,train_labels=y.numpy(),dev_labels=yd.numpy())
    readout_diagnostics=[]
    with np.load(feature_path) as cached:
        for stage in ['initial','final']:
            tx=torch.from_numpy(cached[stage+'_train']);dx=torch.from_numpy(cached[stage+'_dev'])
            fitted=fit_binary_readout(tx,y,ridge=.01)
            def head_score(values,targets):
                predictions=binary_logits(values,fitted)
                return dict(nll=float(F.cross_entropy(predictions,targets)),
                    accuracy=float((predictions.argmax(-1)==targets).double().mean()))
            readout_diagnostics.append(dict(stage=stage,train=head_score(tx,y),dev=head_score(dx,yd),
                fixed_ridge=.01,train_stationarity=fitted['max_gradient']))
    sources=source_hashes()
    for f in ['experiments/hierarchy/frame_learning.py','experiments/hierarchy/part_combination_task.py']:sources[f]=sha(ROOT/f)
    result=dict(status='completed',battle='G-HIER',stage='frame_composition_learning',initial=initial,final=final,
        intervention=intervention,convex_readout_diagnostics=readout_diagnostics,feature_packet=dict(path=str(feature_path.relative_to(ROOT)),sha256=sha(feature_path)),steps=a.steps,history=history,training_scenes=len(train),calibration_scenes=len(calibration_rows),
        dev_scenes=len(dev),parameters=sum(p.numel() for p in model.parameters()),work_training=work_train,
        work_calibration=cal_work,work_dev=dev_work,source_sha256=sources,input_sha256={str(old.relative_to(ROOT)):sha(old)},
        max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,process_wall_s=past_wall+time.monotonic()-started,
        final_weights=str(final_weights.relative_to(ROOT)),scope='Stationary synthetic development; one training seed and common scoring tape. TRAIN-only calibration frozen for unseen DEV combinations/nuisances. TEST not model-scored. No benchmark/family verdict or semantic minimal-depth proof.',
        decision='DEV transfer and emitted-content intervention guide part-binding/credit repair versus larger composition acquisition; no automatic unchanged depth grid.')
    with (ROOT/f'experiments/results/hierarchy/{a.tag}.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(status='completed',final=final,wall_s=result['process_wall_s'])),flush=True)
if __name__=='__main__':main()
