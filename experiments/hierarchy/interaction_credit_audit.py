"""Frozen TRAIN-only native interaction/credit audit; no optimizer or DEV access."""
import argparse,json,sys,time,resource
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch
from torch.nn import functional as F
from experiments.hierarchy.paired_composition_learning import make_model
from experiments.hierarchy.part_combination_task import corpus
from experiments.hierarchy.observed_training_pairs import training_pairs
from experiments.hierarchy.frame_learning import forward
from experiments.hierarchy.native_composition_data import sha


def energy_record(value):
    gradient=value.grad
    return dict(value_norm=float(value.detach().norm()),gradient_norm=0. if gradient is None else float(gradient.norm()),
                sensitivity=0. if gradient is None else float((gradient*value.detach()).norm()))


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args()
    torch.set_num_threads(1);started=time.monotonic();model=make_model('relational');model.train()
    parent='experiments/results/hierarchy/aws_g_hier_paired_relational_20261015_v1.raw.pt'
    model.load_state_dict(torch.load(ROOT/parent,map_location='cpu',weights_only=True))
    train=[r for r in corpus() if r.split=='train'];pairs=training_pairs([r.model_input() for r in train])
    captured={};handles=[]
    def hook(level,kind):
        def save(module,args,out):
            out.retain_grad();captured.setdefault((level,kind),[]).append(out)
        return save
    for level,obj in enumerate(model.objects):
        handles.append(obj.core.content.register_forward_hook(hook(level,'native_input_projection')))
        if level:handles.append(obj.core.composer.register_forward_hook(hook(level,'role_interaction')))
    samples=[];work={}
    # Eight evenly spaced public-input pairs, chosen before observing audit.
    for pair_index in range(0,len(pairs),len(pairs)//8):
        tape=torch.get_rng_state();grads=[]
        for index in pairs[pair_index]:
            torch.set_rng_state(tape);captured.clear();model.zero_grad(set_to_none=True)
            logits,counts=forward(model,train[index]);loss=F.cross_entropy(logits[None],torch.tensor([train[index].label]));loss.backward()
            records={f'{level}:{kind}':[energy_record(v) for v in vals] for (level,kind),vals in captured.items()}
            gradients=[obj.core.composer.pair_weight.grad.detach().flatten().clone() if obj.core.composer.pair_weight.grad is not None else torch.zeros_like(obj.core.composer.pair_weight).flatten() for obj in model.objects[1:]]
            grads.append(gradients);samples.append(dict(pair=pair_index,index=index,loss=float(loss.detach()),layers=records))
            for k,v in counts.items():
                if isinstance(v,int):work[k]=work.get(k,0)+v
            captured.clear();del logits,loss
        cancellation=[]
        for x,y in zip(*grads):
            cancellation.append(float((x+y).norm()/(x.norm()+y.norm()+1e-30)))
        samples[-1]['paired_gradient_retention_by_parent_level']=cancellation
        print(json.dumps(dict(pair=pair_index,gradient_retention=cancellation)),flush=True)
    for h in handles:h.remove()
    result=dict(status='completed',battle='G-HIER',stage='interaction_credit_audit',samples=samples,work=work,process_wall_s=time.monotonic()-started,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        input_sha256={parent:sha(ROOT/parent)},source_sha256={f:sha(ROOT/f) for f in ['experiments/hierarchy/interaction_credit_audit.py','sleeping_machines/relational_frame_composition.py','experiments/hierarchy/paired_composition_learning.py']},
        scope='Sixteen TRAIN scenes/eight observed-input pairs;frozen acquired joint head and native stochastic teacher;no fitting/DEV/TEST. Gradient-times-value is local first-order sensitivity,not causal attribution or gradient correctness proof.')
    path=ROOT/f'experiments/results/hierarchy/{a.tag}.json'
    with path.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
if __name__=='__main__':main()
