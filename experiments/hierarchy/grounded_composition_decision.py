"""Frozen three-arm representation versus decoder decision; TEST excluded."""
import argparse,json,math,resource,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from experiments.hierarchy.carried_role_decision import observed_groups,interval,apply_emission_cut
from experiments.hierarchy.grounded_composition_learning import make_model,score
from experiments.hierarchy.part_combination_task import corpus
from experiments.hierarchy.native_composition_data import sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args();torch.set_num_threads(1);started=time.monotonic()
    dev=[r for r in corpus() if r.split=='dev'];labels=torch.tensor([r.label for r in dev]);groups=observed_groups([r.model_input() for r in dev])
    rows={};losses={};bindings={};interventions={}
    for arm in ['joint','continued','decoder_only']:
        path=ROOT/f'experiments/results/hierarchy/aws_g_hier_grounded_{arm}_20261016_v1.json';fit=json.loads(path.read_text());assert fit['status']=='completed'
        feature=ROOT/fit['feature_packet']['path'];assert sha(feature)==fit['feature_packet']['sha256']
        weights=ROOT/fit['final_weights'];state=torch.load(weights,map_location='cpu',weights_only=True)
        with np.load(feature) as packet:
            assert np.array_equal(packet['dev_labels'],labels.numpy())
            x=torch.from_numpy(packet['final_dev']);change=float(np.max(np.abs(packet['final_dev']-packet['initial_dev'])))
            if arm=='decoder_only':assert change==0
        logits=F.linear(x,state['head.weight'],state['head.bias']);loss=F.cross_entropy(logits,labels,reduction='none').numpy();losses[arm]=loss
        assert abs(float(loss.mean())-fit['final']['materialized_dev_nll'])<1e-6
        rows[arm]=dict(nll=float(loss.mean()),accuracy=float((logits.argmax(-1)==labels).double().mean()),benefit_over_prior=interval(math.log(2)-loss,groups),parameters=fit['parameters'],process_wall_s=fit['process_wall_s'],work_training=fit['work_training'],work_dev=fit['work_final_dev'],feature_change=change,grounding_dev=fit['final']['grounding_dev'])
        for q in [path,feature,weights]:bindings[str(q.relative_to(ROOT))]=sha(q)
        if arm=='joint' and float(loss.mean())<math.log(2):
            interventions[arm]={}
            for cut in ['values','child_delays','values_and_child_delays']:
                model=make_model(arm);model.load_state_dict(state);apply_emission_cut(model,cut)
                changed,_,work=score(model,dev);altered=F.cross_entropy(changed,labels,reduction='none').numpy()
                interventions[arm][cut]=dict(nll=float(altered.mean()),mechanism_benefit=interval(altered-loss,groups),work=work)
    gain=interval(losses['continued']-losses['joint'],groups)
    transfer=rows['joint']['benefit_over_prior']['bootstrap_95'][0]>0
    beats_control=gain['bootstrap_95'][0]>0
    mechanism=any(x['mechanism_benefit']['bootstrap_95'][0]>0 for x in interventions.get('joint',{}).values())
    selected='joint' if transfer and beats_control and mechanism else None
    decision='Grounding improves native development transfer with emitted-channel evidence;retain for fresh independent composition confirmation.' if selected else 'Grounding has no qualified transfer gain against continued native learning. Compare reconstruction with frozen-encoder control and feature changes before another fit;no automatic scaling.'
    result=dict(status='completed',battle='G-HIER',stage='grounded_composition_decision',rows=rows,joint_nll_gain=gain,interventions=interventions,selected_for_confirmation=selected,decision=decision,input_sha256=bindings,source_sha256={f:sha(ROOT/f) for f in ['experiments/hierarchy/grounded_composition_decision.py','experiments/hierarchy/grounded_composition_learning.py','sleeping_machines/grounded_native_composition.py']},process_wall_s=time.monotonic()-started,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scope='One native seed,TRAIN-only heads,64reused synthetic DEV scenes/16observed geometry clusters,128new presentations per arm,different learning work. Fixed geometry reconstruction is a decoder diagnostic,not semantic sufficiency. TEST not model-scored.')
    with (ROOT/f'experiments/results/hierarchy/{a.tag}.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(decision=decision,rows={k:v['nll'] for k,v in rows.items()})),flush=True)
if __name__=='__main__':main()
