"""Frozen TRAIN-only native credit audit;no ablation,parameter update or fit.

Reports the implemented counterfactual teacher's linear VJP,not an exact
Jacobian of discontinuous expected hard-routing loss. Readout derivatives
are ordinary exact derivatives. Uses the actual jointly trained head.
"""
import argparse,collections,json,math,resource,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch
from torch.nn import functional as F
from experiments.hierarchy.grounded_composition_learning import scene_forward
from experiments.hierarchy.part_combination_task import corpus
from experiments.hierarchy.observed_training_pairs import training_pairs
from experiments.hierarchy.native_composition_data import sha,source_hashes
from sleeping_machines.part_identity_composition import fit_shape_vocabulary
from sleeping_machines.identity_role_composition import IdentityRoleComposition
BASE='experiments/results/hierarchy/aws_g_hier_composition_scale_20261016_v1'
CHECKPOINTS={step:BASE+f'.step{step}.pt' for step in [128,512]}
AUDIT='experiments/results/hierarchy/aws_g_hier_ownership_support_20261016_v1.json'


def block(name):
 if name.startswith('head.'):return 'readout'
 if name.startswith('identity_head.'):return 'local_identity_head'
 if name.startswith('objects.'):
  level=name.split('.')[1];relation=('.composer.' in name or name.endswith(('.same_role_weight','.relative_weight')))
  return 'level_'+level+('_relation' if relation else '_native')
 return 'input_interface'


def pair_summary(ga,gb,alpha,beta):
 common=(ga+gb)*.5;odd=(ga-gb)*.5
 g=(alpha*ga+beta*gb)*.5
 reconstruction=((alpha+beta)*common+(alpha-beta)*odd)*.5
 denom=(abs(alpha)*ga.norm()+abs(beta)*gb.norm())*.5
 return dict(credit_norm=float(g.norm()),individual_credit_mean=float(denom),cancellation_ratio=float(g.norm()/denom) if denom>0 else None,logit_vjp_norm_mean=float((ga.norm()+gb.norm())*.5),odd_vjp_norm=float(odd.norm()),odd_to_total=float(odd.norm()/((ga.norm()+gb.norm())*.5)) if ga.norm()+gb.norm()>0 else None,decomposition_abs_error=float((g-reconstruction).abs().max()) if len(g) else 0.)


def main():
 p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args();began=time.monotonic();torch.set_num_threads(1)
 rows=[r for r in corpus() if r.split=='train' and r.nuisance==0];pairs=training_pairs([r.model_input() for r in rows]);indices=[round(k*(len(pairs)-1)/7) for k in range(8)]
 threshold=json.loads((ROOT/AUDIT).read_text())['threshold'];vocab=fit_shape_vocabulary([r.events for r in rows],threshold)
 torch.manual_seed(1451);model=IdentityRoleComposition(classes=2,ownership_threshold=threshold,shape_vocabulary=vocab);model.train()
 params=list(model.named_parameters());names=[n for n,p in params];leaves=[p for n,p in params]
 results=[];work={};max_linearity=0.;input_hashes={AUDIT:sha(ROOT/AUDIT)}
 for step,path in CHECKPOINTS.items():
  snapshot=torch.load(ROOT/path,map_location='cpu',weights_only=False);model.load_state_dict(snapshot['model']);input_hashes[path]=sha(ROOT/path);records=[]
  for index in indices:
   samples=[]
   for row_index in pairs[index]:
    row=rows[row_index];torch.manual_seed(113);captured=[]
    hook=model.head.register_forward_pre_hook(lambda module,args:captured.append(args[0]))
    try:z,state,_=scene_forward(model,row)
    finally:hook.remove()
    d=z[1]-z[0];loss=F.cross_entropy(z[None],torch.tensor([row.label]));alpha=torch.sigmoid(d.detach())-row.label
    vjp=torch.autograd.grad(d,leaves,retain_graph=True,allow_unused=True)
    ce=torch.autograd.grad(loss,leaves,allow_unused=True)
    grouped=collections.defaultdict(list);linear_error=0.
    for name,param,grad,actual in zip(names,leaves,vjp,ce):
     g=torch.zeros_like(param) if grad is None else grad.detach();true=torch.zeros_like(param) if actual is None else actual.detach()
     linear_error=max(linear_error,float((true-alpha*g).abs().max()));grouped[block(name)].append(g.reshape(-1).double())
    max_linearity=max(max_linearity,linear_error)
    assert linear_error<2e-5,'implemented VJP lost scalar chain rule'
    samples.append(dict(label=row.label,log_odds=float(d.detach()),probability=float(torch.sigmoid(d.detach())),loss=float(loss.detach()),alpha=float(alpha),feature=captured[0].detach().double(),grads={k:torch.cat(v) for k,v in grouped.items()},input_digest=row.digest,linearity_abs_error=linear_error))
    for k,v in model.work(state).items():
     if isinstance(v,int):work[k]=work.get(k,0)+v
    del state,z,d,loss,vjp,ce,captured
   x,y=samples;groups={k:pair_summary(x['grads'][k],y['grads'][k],x['alpha'],y['alpha']) for k in x['grads']}
   # Pure readout has an exact analytically known derivative to log odds.
   feature_difference=float((x['feature']-y['feature']).norm())
   records.append(dict(pair_index=index,input_digests=[x['input_digest'],y['input_digest']],labels=[x['label'],y['label']],log_odds=[x['log_odds'],y['log_odds']],logit_contrast=x['log_odds']-y['log_odds'],root_feature_distance=feature_difference,pair_loss=(x['loss']+y['loss'])/2,groups=groups))
  keys=records[0]['groups'];summary={k:{metric:sum(r['groups'][k][metric] for r in records if r['groups'][k][metric] is not None)/sum(r['groups'][k][metric] is not None for r in records) if any(r['groups'][k][metric] is not None for r in records) else None for metric in records[0]['groups'][k]} for k in keys}
  results.append(dict(additional_updates=step,pairs=records,group_means=summary,mean_root_feature_distance=sum(r['root_feature_distance'] for r in records)/len(records),mean_absolute_logit_contrast=sum(abs(r['logit_contrast']) for r in records)/len(records),joint_head_weight_difference=float((model.head.weight[1]-model.head.weight[0]).norm())))
  print(json.dumps(dict(step=step,root_feature_distance=results[-1]['mean_root_feature_distance'],logit_contrast=results[-1]['mean_absolute_logit_contrast'],readout_cancellation=summary['readout']['cancellation_ratio'])),flush=True)
 sources=source_hashes();sources['experiments/hierarchy/composition_credit_audit.py']=sha(Path(__file__))
 result=dict(status='completed',battle='G-HIER',stage='composition_credit_audit',rows=results,max_scalar_chain_rule_error=max_linearity,work=work,process_wall_s=time.monotonic()-began,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,source_sha256=sources,input_sha256=input_hashes,scope='Frozen128/512update snapshots,eight original TRAIN pairs/common race tape113,actual joint head and native counterfactual teacher.No DEV/TEST,no ablation,no optimizer update or fit.Credit VJPs include the implemented local routing surrogate,not exact expected-loss Jacobians;readout VJPs are exact. Paired cancellation and sensitivity are local evidence,not causal proof of a unique failure.')
 out=ROOT/f'experiments/results/hierarchy/{a.tag}.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
