"""Separate native composition representation learning from readout optimization.

Fixed TRAIN-only ridge logistic head; acquired initial and completed final cores.
No native backbone fitting or TEST model scoring. Cache features for reuse.
"""
import argparse,json,math,resource,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from experiments.hierarchy.part_combination_task import corpus
from experiments.hierarchy.coalesced_learning import forward,ZeroEmittedContent
from experiments.hierarchy.native_composition_data import sha,source_hashes
from sleeping_machines.coalesced_native_composition import CoalescedNativeComposition
from sleeping_machines.convex_native_readout import fit_binary_readout,binary_logits


def features(model,rows):
    model.eval();values=[];counts={}
    hook=model.head.register_forward_pre_hook(lambda module,args:values.append(args[0].detach().clone()))
    try:
        with torch.no_grad():
            for row in rows:
                torch.manual_seed(113);_,work=forward(model,row)
                for key,val in work.items():
                    if isinstance(val,int):counts[key]=counts.get(key,0)+val
    finally:hook.remove()
    return torch.stack(values),torch.tensor([r.label for r in rows]),counts


def summary(x,y,fit):
    z=binary_logits(x,fit);loss=F.cross_entropy(z,y,reduction='none')
    return dict(nll=float(loss.mean()),accuracy=float((z.argmax(-1)==y).double().mean()),losses=loss.tolist())


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args()
    torch.set_num_threads(1);started=time.monotonic()
    fit_path=ROOT/'experiments/results/hierarchy/aws_g_hier_coalesced_learning_20261015_v1.json'
    completed=json.loads(fit_path.read_text());assert completed['status']=='completed'
    rows=corpus();train=[r for r in rows if r.split=='train' and r.nuisance==0];dev=[r for r in rows if r.split=='dev']
    torch.manual_seed(211);initial=CoalescedNativeComposition(classes=2)
    old=ROOT/'experiments/results/hierarchy/aws_g_hier_native_composition_20261015_v1_learn.final.pt'
    initial.load_state_dict({k:v for k,v in torch.load(old,map_location='cpu',weights_only=True).items() if not k.startswith('head.')},strict=False)
    final=CoalescedNativeComposition(classes=2);final_path=ROOT/completed['final_weights']
    final.load_state_dict(torch.load(final_path,map_location='cpu',weights_only=True))
    packet={};records=[];final_fit=None
    for label,model in [('initial',initial),('final',final)]:
        x,y,wt=features(model,train);xd,yd,wd=features(model,dev)
        fitted=fit_binary_readout(x,y,ridge=.01)
        records.append(dict(variant=label,train=summary(x,y,fitted),dev=summary(xd,yd,fitted),
             fit={k:v for k,v in fitted.items() if isinstance(v,(int,float))},work_train=wt,work_dev=wd,
             feature_std_mean=float(x.double().std(0,unbiased=False).mean())))
        packet.update({label+'_train':x.numpy(),label+'_dev':xd.numpy()})
        if label=='final':final_fit=fitted
    intervention=None
    if records[-1]['dev']['nll']<math.log(2):
        altered=ZeroEmittedContent(classes=2);altered.load_state_dict(final.state_dict())
        xa,ya,wa=features(altered,dev);intervention=summary(xa,ya,final_fit);intervention['work']=wa
        packet['zero_emitted_dev']=xa.numpy()
        delta=np.array(intervention['losses'])-np.array(records[-1]['dev']['losses'])
        groups={}
        for i,r in enumerate(dev):groups.setdefault((r.left,r.right),[]).append(i)
        means=np.array([delta[idx].mean() for idx in groups.values()]);rng=np.random.default_rng(229)
        boot=means[rng.integers(len(means),size=(2000,len(means)))].mean(1)
        intervention['paired_nll_gain']=float(means.mean());intervention['combination_bootstrap_95']=np.quantile(boot,[.025,.975]).tolist()
    weights={k:v.detach().clone() for k,v in final.state_dict().items()}
    with torch.no_grad():
        final.head.weight.zero_();final.head.bias.zero_()
        final.head.weight[1].copy_(final_fit['weight']);final.head.bias[1].copy_(final_fit['bias'])
    assert all(torch.equal(value,final.state_dict()[key]) for key,value in weights.items() if not key.startswith('head.'))
    out=ROOT/f'experiments/results/hierarchy/{a.tag}.json'
    weight_path=out.with_suffix('.readout.pt');torch.save(final.state_dict(),weight_path)
    feature_path=out.with_name(out.stem+'_features.npz')
    np.savez_compressed(feature_path,**packet,train_labels=y.numpy(),dev_labels=yd.numpy())
    sources=source_hashes()
    for f in ['experiments/hierarchy/coalesced_frozen_readout.py','experiments/hierarchy/coalesced_learning.py','experiments/hierarchy/part_combination_task.py','sleeping_machines/convex_native_readout.py']:sources[f]=sha(ROOT/f)
    result=dict(status='completed',battle='G-HIER',stage='coalesced_frozen_readout',rows=records,intervention=intervention,
      source_sha256=sources,input_sha256={str(p.relative_to(ROOT)):sha(p) for p in [fit_path,old,final_path]},
      feature_packet=dict(path=str(feature_path.relative_to(ROOT)),sha256=sha(feature_path)),unchanged_native_parameters=True,
      final_weights=str(weight_path.relative_to(ROOT)),max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
      process_wall_s=time.monotonic()-started,
      scope='Synthetic development, same64 TRAIN and64 unseen-combination DEV scenes, common noise tape. Fixed ridge.01; no DEV hyperparameter selection, native refit or TEST model scoring.',
      decision='Initial/final frozen head fits distinguish readout fitting from learned feature gain; DEV plus emitted-content intervention guides binding/credit versus backbone optimization repair.')
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(status='completed',rows=[dict(variant=r['variant'],train_nll=r['train']['nll'],dev_nll=r['dev']['nll']) for r in records])),flush=True)
if __name__=='__main__':main()
