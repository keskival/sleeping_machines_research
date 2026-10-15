"""Reuse native pair endpoints/caches; conditionally audit emitted content.

No new fitting or TEST model evaluation. Resampling groups use observed geometry
only, not hidden generator identities.
"""
import argparse,hashlib,json,math,resource,sys,time
from dataclasses import replace
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from experiments.hierarchy.part_combination_task import corpus
from experiments.hierarchy.observed_training_pairs import swap_observed_halves
from experiments.hierarchy.paired_composition_learning import make_model
from experiments.hierarchy.frame_learning import score
from experiments.hierarchy.native_composition_data import sha,source_hashes


def observed_groups(examples):
    def signature(events):
        xy=events[:,1:3]-events[:,1:3].min(0)
        geometry=np.column_stack((xy,events[:,3]))
        ordered=np.array(sorted(map(tuple,geometry)),dtype=np.int64)
        return hashlib.sha256(ordered.tobytes()).hexdigest()
    groups={}
    for i,(events,_) in enumerate(examples):
        key=min(signature(events),signature(swap_observed_halves(events)))
        groups.setdefault(key,[]).append(i)
    return list(groups.values())


def interval(delta,groups):
    means=np.array([np.asarray(delta)[idx].mean() for idx in groups]);rng=np.random.default_rng(229)
    boot=means[rng.integers(len(means),size=(2000,len(means)))].mean(1)
    return dict(mean=float(means.mean()),bootstrap_95=np.quantile(boot,[.025,.975]).tolist(),groups=len(groups))


def apply_emission_cut(model,cut):
    if cut not in ['values','child_delays','values_and_child_delays']:raise ValueError('unknown emission intervention')
    if cut in ['values','values_and_child_delays']:
        original=model._send
        def zero_emitted(state,source_level,source_address,value,version,time,kind,target=None):
            if source_level>=0 and kind=='timer':value=value*0
            return original(state,source_level,source_address,value,version,time,kind,target)
        model._send=zero_emitted
    if cut in ['child_delays','values_and_child_delays']:
        for obj in model.objects[:-1]:
            original_arm=obj.arm
            def fixed_arm(state,address,now,head=0,_original=original_arm):
                timer=_original(state,address,now,head)
                due=torch.as_tensor(now,dtype=timer.due.dtype,device=timer.due.device)+.011
                return replace(timer,due=due)
            obj.arm=fixed_arm
    return model


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args()
    torch.set_num_threads(1);started=time.monotonic();dev=[r for r in corpus() if r.split=='dev']
    examples=[r.model_input() for r in dev];groups=observed_groups(examples)
    labels=torch.tensor([y for _,y in examples]);records={};losses={};inputs={};interventions={}
    for arm in ['frame','relational']:
        path=ROOT/f'experiments/results/hierarchy/aws_g_hier_paired_{arm}_20261015_v1.json'
        fit=json.loads(path.read_text());assert fit['status']=='completed' and fit['arm']==arm
        feature=ROOT/fit['feature_packet']['path'];assert sha(feature)==fit['feature_packet']['sha256']
        weights=ROOT/fit['final_weights'];state=torch.load(weights,map_location='cpu',weights_only=True)
        with np.load(feature) as cached:
            assert np.array_equal(cached['dev_labels'],labels.numpy());x=torch.from_numpy(cached['final_dev'])
        logits=F.linear(x,state['head.weight'],state['head.bias'])
        loss=F.cross_entropy(logits,labels,reduction='none').numpy();losses[arm]=loss
        assert abs(float(loss.mean())-fit['final']['materialized_dev_nll'])<1e-6
        benefit=interval(math.log(2)-loss,groups)
        records[arm]=dict(nll=float(loss.mean()),accuracy=float((logits.argmax(-1)==labels).double().mean()),
          benefit_over_prior=benefit,initial_dev_nll=fit['initial']['primary_dev']['nll'],
          parameters=fit['parameters'],process_wall_s=fit['process_wall_s'],work_training=fit['work_training'],work_dev=fit['work_final_dev'])
        inputs.update({str(q.relative_to(ROOT)):sha(q) for q in [path,feature,weights]})
        if float(loss.mean())<math.log(2):
            cuts={}
            for cut in ['values','child_delays','values_and_child_delays']:
                model=make_model(arm);model.load_state_dict(state)
                apply_emission_cut(model,cut)
                altered,_,work=score(model,dev)
                changed=F.cross_entropy(altered,labels,reduction='none').numpy()
                cuts[cut]=dict(nll=float(changed.mean()),mechanism_benefit=interval(changed-loss,groups),work=work)
                del model
            interventions[arm]=cuts
    comparison=interval(losses['frame']-losses['relational'],groups)
    transfer=[arm for arm in records if records[arm]['benefit_over_prior']['bootstrap_95'][0]>0]
    promising=[arm for arm in transfer if arm in interventions and any(
       cut['mechanism_benefit']['bootstrap_95'][0]>0 for cut in interventions[arm].values())]
    if promising:
        decision='Positive development transfer with emitted-value/timing mechanism evidence: retain the best qualified arm for fresh independent composition confirmation.'
    elif transfer:
        decision='Positive DEV prediction; value/delay cuts do not attribute the benefit. Investigate address/availability and internal timing before semantic promotion;no larger fit automatically.'
    else:
        decision='No arm clears the transfer gate. Reuse cached features and diagnosed interaction/support/credit paths;no unchanged larger fit or depth sweep.'
    sources=source_hashes()
    for f in ['experiments/hierarchy/paired_composition_decision.py','experiments/hierarchy/observed_training_pairs.py','experiments/hierarchy/paired_composition_learning.py','experiments/hierarchy/frame_learning.py','experiments/hierarchy/part_combination_task.py']:sources[f]=sha(ROOT/f)
    result=dict(status='completed',battle='G-HIER',stage='paired_composition_decision',rows=records,
       relational_nll_gain=comparison,interventions=interventions,transfer_positive_arms=transfer,qualified_arms=promising,selected_for_confirmation=min(promising,key=lambda name:records[name]['nll']) if promising else None,decision=decision,
       input_sha256=inputs,source_sha256=sources,process_wall_s=time.monotonic()-started,
       max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
       intervention_scope='Values cut retains clocks/addresses/availability. Child-delay cut fixes arm-to-emit delay at .011,retains random draws,reads views later and changes transport/scheduling jointly. Root timer unchanged;not a cut of all timing/availability paths.',
       scope='One-seed synthetic DEV comparison,TRAIN-only heads,observed-geometry clusters,common noise tape. Same128presentations;different parameters/arithmetic. Reused DEV is not sealed confirmation. TEST not model-scored.')
    with (ROOT/f'experiments/results/hierarchy/{a.tag}.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(rows={k:dict(nll=v['nll'],accuracy=v['accuracy']) for k,v in records.items()},decision=decision)),flush=True)
if __name__=='__main__':main()
