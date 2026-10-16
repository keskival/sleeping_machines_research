"""G-HIER: paired whole-label native composition development, no TEST scoring."""
import argparse,json,math,resource,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from experiments.hierarchy.part_combination_task import corpus,audit
from experiments.hierarchy.observed_training_pairs import training_pairs
from experiments.hierarchy.frame_learning import score,forward,atomic_save
from experiments.hierarchy.native_composition_data import sha,source_hashes
from sleeping_machines.grounded_native_composition import GroundedNativeComposition
from sleeping_machines.frame_native_composition import FrameNativeComposition
from sleeping_machines.relational_frame_composition import RelationalFrameComposition
from sleeping_machines.convex_native_readout import fit_binary_readout,binary_logits
from sleeping_machines.prior_centered_readout import fit_prior_centered
PARENT=ROOT/'experiments/results/hierarchy/aws_g_hier_carried_continued_20261015_v1.raw.pt'


def make_model(arm):
    if arm not in ['joint','continued','decoder_only']:raise ValueError('unknown arm')
    torch.manual_seed(941);model=GroundedNativeComposition(classes=2)
    model.load_relational_state(torch.load(PARENT,map_location='cpu',weights_only=True))
    if arm=='decoder_only':
        for name,p in model.named_parameters():p.requires_grad_(name.startswith(('geometry_decoders.','head.')))
    model.learning_arm=arm
    return model


def scene_forward(model,row):
    events,label=row.model_input();state=model.new_state()
    for micros,y,x,pol in events:model.observe(state,(int(y),int(x)),int(micros)/1e6,[float(pol),1.])
    z,_=model.compose(state);seconds=events.astype('float64');seconds[:,0]/=1e6
    return z,state,seconds


def score(model,rows):
    model.eval();tape=torch.get_rng_state();features=[];zs=[];work={};local=[]
    hook=model.head.register_forward_pre_hook(lambda module,args:features.append(args[0].detach().clone()))
    try:
        with torch.no_grad():
            for row in rows:
                torch.manual_seed(113);z,state,events=scene_forward(model,row)
                _,record=model.grounding_loss(state,events)
                local.append([float(x) for x in record['level_losses']]);zs.append(z)
                for k,v in model.work(state).items():
                    if isinstance(v,int):work[k]=work.get(k,0)+v
    finally:hook.remove();torch.set_rng_state(tape)
    model.scored_features=torch.stack(features);model.scored_grounding=np.asarray(local).mean(0).tolist()
    return torch.stack(zs),torch.tensor([r.label for r in rows]),work


def paired_backward(model,examples):
    tape=torch.get_rng_state();next_tape=None;losses=[];counts={}
    for j,(row,label) in enumerate(examples):
        if j:torch.set_rng_state(tape)
        logits,state,events=scene_forward(model,row);loss=F.cross_entropy(logits[None],torch.tensor([label]))
        if model.learning_arm!='continued':
            local,_=model.grounding_loss(state,events);loss=loss+.1*local
        (loss/2).backward();losses.append(float(loss.detach()))
        for key,val in model.work(state).items():
            if isinstance(val,int):counts[key]=counts.get(key,0)+val
        if not j:next_tape=torch.get_rng_state()
        del logits,loss,state
    torch.set_rng_state(next_tape)
    return sum(losses)/2,counts


def endpoint(z,y,features,zd,yd,dev_features):
    cal=fit_prior_centered(z,y);fit=fit_binary_readout(features,y,ridge=.01)
    def metrics(predictions,targets):
        losses=F.cross_entropy(predictions,targets,reduction='none')
        return dict(nll=float(losses.mean()),accuracy=float((predictions.argmax(-1)==targets).double().mean()),losses=losses.tolist())
    return dict(primary_train=metrics(binary_logits(features,fit),y),primary_dev=metrics(binary_logits(dev_features,fit),yd),
        joint_train=metrics(z,y),joint_dev=metrics(zd,yd),
        scalar_dev=metrics(cal['log_prior']+cal['gain']*(zd.double()-cal['center']),yd),
        scalar_gain=cal['gain'],fit_gradient=fit['max_gradient']),fit


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);p.add_argument('--arm',choices=['joint','continued','decoder_only'],required=True);p.add_argument('--steps',type=int,default=64)
    a=p.parse_args();torch.set_num_threads(1);started=time.monotonic();rows=corpus();audit(rows)
    train=[r for r in rows if r.split=='train'];calibration=[r for r in train if r.nuisance==0];dev=[r for r in rows if r.split=='dev']
    pairs=training_pairs([r.model_input() for r in train]);model=make_model(a.arm)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001,weight_decay=.01);order=torch.Generator().manual_seed(953)
    prefix=ROOT/f'experiments/results/hierarchy/{a.tag}';checkpoint=prefix.with_suffix('.progress.pt');initial_file=prefix.with_suffix('.initial_features.npz')
    start=0;history=[];initial=None;work_train={};past=0.
    if checkpoint.exists():
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False);assert saved['arm']==a.arm and saved['steps']==a.steps
        assert sha(initial_file)==saved['initial_sha256']
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);torch.set_rng_state(saved['rng']);order.set_state(saved['order_rng'])
        start=saved['step'];history=saved['history'];initial=saved['initial'];work_train=saved['work_train'];past=saved['wall_s']
    if initial is None:
        z,y,wt=score(model,calibration);x=model.scored_features.clone();train_grounding=model.scored_grounding.copy()
        zd,yd,wd=score(model,dev);xd=model.scored_features.clone();dev_grounding=model.scored_grounding.copy()
        initial,_=endpoint(z,y,x,zd,yd,xd);initial.update(work_calibration=wt,work_dev=wd,grounding_train=train_grounding,grounding_dev=dev_grounding)
        np.savez_compressed(initial_file,train=x.numpy(),dev=xd.numpy())
    for step in range(start,a.steps):
        i,j=pairs[int(torch.randint(len(pairs),(),generator=order))]
        model.train(a.arm!='decoder_only');optimizer.zero_grad(set_to_none=True)
        loss,counts=paired_backward(model,[(train[i],train[i].label),(train[j],train[j].label)])
        assert math.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();history.append(loss)
        for key,val in counts.items():work_train[key]=work_train.get(key,0)+val
        if (step+1)%8==0 or step+1==a.steps:
            atomic_save(checkpoint,dict(arm=a.arm,steps=a.steps,step=step+1,model=model.state_dict(),optimizer=optimizer.state_dict(),rng=torch.get_rng_state(),order_rng=order.get_state(),history=history,initial=initial,initial_sha256=sha(initial_file),work_train=work_train,wall_s=past+time.monotonic()-started))
            print(json.dumps(dict(step=step+1,paired_last8=sum(history[-8:])/len(history[-8:]))),flush=True)
    z,y,wt=score(model,calibration);x=model.scored_features.clone();train_grounding=model.scored_grounding.copy()
    zd,yd,wd=score(model,dev);xd=model.scored_features.clone();dev_grounding=model.scored_grounding.copy()
    final,fitted=endpoint(z,y,x,zd,yd,xd);final.update(grounding_train=train_grounding,grounding_dev=dev_grounding)
    raw_file=prefix.with_suffix('.raw.pt');atomic_save(raw_file,model.state_dict())
    with torch.no_grad():
        model.head.weight.zero_();model.head.bias.zero_();model.head.weight[1].copy_(fitted['weight']);model.head.bias[1].copy_(fitted['bias'])
    # Score the actual float32 saved head, not only the float64 solver.
    actual=F.linear(x,model.head.weight,model.head.bias);actual_dev=F.linear(xd,model.head.weight,model.head.bias)
    final['materialized_train_nll']=float(F.cross_entropy(actual,y));final['materialized_dev_nll']=float(F.cross_entropy(actual_dev,yd))
    final_file=prefix.with_suffix('.final.pt');atomic_save(final_file,model.state_dict())
    packet=prefix.with_name(prefix.name+'_features.npz')
    with np.load(initial_file) as old:np.savez_compressed(packet,initial_train=old['train'],initial_dev=old['dev'],final_train=x.numpy(),final_dev=xd.numpy(),train_labels=y.numpy(),dev_labels=yd.numpy())
    sources=source_hashes()
    for f in ['experiments/hierarchy/grounded_composition_learning.py','sleeping_machines/grounded_native_composition.py','sleeping_machines/geometry_sketch_algebra.py','experiments/hierarchy/observed_training_pairs.py','experiments/hierarchy/part_combination_task.py','experiments/hierarchy/frame_learning.py']:sources[f]=sha(ROOT/f)
    result=dict(status='completed',battle='G-HIER',stage='grounded_composition_learning',arm=a.arm,steps=a.steps,presentations=2*a.steps,
        parameters=sum(p.numel() for p in model.parameters()),initial=initial,final=final,history=history,
        work_training=work_train,work_final_calibration=wt,work_final_dev=wd,source_sha256=sources,
        input_sha256={str(PARENT.relative_to(ROOT)):sha(PARENT)},final_weights=str(final_file.relative_to(ROOT)),raw_weights=str(raw_file.relative_to(ROOT)),
        feature_packet=dict(path=str(packet.relative_to(ROOT)),sha256=sha(packet)),process_wall_s=past+time.monotonic()-started,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scope='One seed,acquired continued-relational actor;paired TRAIN labels plus fixed observable local geometry,lambda0.1 for joint/frozen-decoder arms. Continued arm CE-only. Decoder-only freezes native encoder and uses selected-only forward;native arms retain counterfactual learning. Reconstructions are decoder diagnostics,not labels or inference inputs. Fixed TRAIN ridge.01 primary affine head,unseen DEV combinations/nuisances,TEST not model-scored. Equal presentations,not equal arithmetic/parameters.',
        decision='Does local compositional grounding improve representations and transfer beyond continued native learning and frozen-encoder decoder learning? DEV success requires frozen-head emitted-content confirmation before promotion; no automatic larger fit.')
    with prefix.with_suffix('.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(arm=a.arm,dev_nll=final['primary_dev']['nll'],dev_accuracy=final['primary_dev']['accuracy'])),flush=True)
if __name__=='__main__':main()
