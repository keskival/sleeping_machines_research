"""TRAIN-only integrated multi-level access and direct-credit update gate."""
import argparse,json,resource,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch
from sleeping_machines.carried_role_composition import CarriedRoleComposition
from sleeping_machines.responsibility_native_composition import ResponsibilityNativeComposition
from experiments.hierarchy.part_combination_task import corpus
from experiments.hierarchy.observed_training_pairs import training_pairs
from experiments.hierarchy.frame_learning import forward
from experiments.hierarchy.native_composition_data import sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args();torch.set_num_threads(1)
    parent='experiments/results/hierarchy/aws_g_hier_paired_relational_20261015_v1.raw.pt'
    torch.manual_seed(211);carrier=CarriedRoleComposition(classes=2);carrier.load_relational_state(torch.load(ROOT/parent,map_location='cpu',weights_only=True))
    torch.manual_seed(211);model=ResponsibilityNativeComposition(classes=2);model.load_carried_state(carrier.state_dict());del carrier
    train=[r for r in corpus() if r.split=='train'];pair=training_pairs([r.model_input() for r in train])[0]
    model.train();optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01);optimizer.zero_grad(set_to_none=True)
    start=time.monotonic();samples=[];work={}
    for index in pair:
        torch.manual_seed(113);logits,counts=forward(model,train[index]);record=model.last_responsibility
        loss,parts=model.responsibility.objective(logits,record,train[index].label)
        (loss/2).backward()
        samples.append(dict(prediction_loss=float(parts['prediction'].detach()),auxiliary_loss=float(parts['auxiliary'].detach()),weights=record['weights'].detach().tolist(),allocation_level=float(record['allocation_level_expectation'].detach())))
        for k,v in counts.items():
            if isinstance(v,int):work[k]=work.get(k,0)+v
        model.last_responsibility=None;del logits,record,loss,parts
    gradients=[float(obj.core.content.weight.grad.norm()) for obj in model.objects]
    assert all(g>0 for g in gradients) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    gate_gradient=model.responsibility.gate.grad.detach().tolist()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
    files=['experiments/hierarchy/responsibility_gate.py','sleeping_machines/responsibility_readout.py','sleeping_machines/responsibility_native_composition.py','experiments/theory/PREDICTIVE_RESPONSIBILITY_TRANSFER.md']
    result=dict(status='completed',battle='G-HIER',stage='responsibility_gate',parameters=sum(p.numel() for p in model.parameters()),paired_update_s=time.monotonic()-start,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,samples=samples,content_gradients=gradients,gate_gradient=gate_gradient,work=work,input_sha256={parent:sha(ROOT/parent)},source_sha256={f:sha(ROOT/f) for f in files},scope='One observed TRAIN pair,whole labels only,one paired AdamW update,full32native stages/eight levels. Output mean summaries are a restricted spatial-access diagnostic. No DEV/TEST,quality,semantic hierarchy or traffic saving claim. All transmissions remain enabled.')
    with (ROOT/f'experiments/results/hierarchy/{a.tag}.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
