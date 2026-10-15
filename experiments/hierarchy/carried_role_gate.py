"""One-pair integrated carrier sensitivity/resource/update gate; TRAIN only."""
import argparse,json,sys,time,resource
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch
from torch.nn import functional as F
from sleeping_machines.carried_role_composition import CarriedRoleComposition
from experiments.hierarchy.paired_composition_learning import make_model
from experiments.hierarchy.part_combination_task import corpus
from experiments.hierarchy.observed_training_pairs import training_pairs
from experiments.hierarchy.frame_learning import forward
from experiments.hierarchy.native_composition_data import sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args();torch.set_num_threads(1)
    parent='experiments/results/hierarchy/aws_g_hier_paired_relational_20261015_v1.raw.pt';weights=torch.load(ROOT/parent,map_location='cpu',weights_only=True)
    train=[r for r in corpus() if r.split=='train'];pair=training_pairs([r.model_input() for r in train])[0];rows=[]
    for arm in ['relational','carried']:
        torch.manual_seed(211)
        model=make_model('relational') if arm=='relational' else CarriedRoleComposition(classes=2)
        if arm=='relational':model.load_state_dict(weights)
        else:model.load_relational_state(weights)
        model.train();optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01);optimizer.zero_grad(set_to_none=True)
        start=time.monotonic();sens=[];counts={}
        for index in pair:
            torch.manual_seed(113);captures={};handles=[]
            for level,obj in enumerate(model.objects[1:],1):
                def capture(module,args,value,level=level):value.retain_grad();captures.setdefault(level,[]).append(value)
                handles.append(obj.core.composer.register_forward_hook(capture))
            z,work=forward(model,train[index]);loss=F.cross_entropy(z[None],torch.tensor([train[index].label]))/2;loss.backward()
            sens.append({str(l):sum(float((v.detach()*v.grad).norm()) for v in vals)/len(vals) for l,vals in captures.items()})
            for h in handles:h.remove()
            for k,v in work.items():
                if isinstance(v,int):counts[k]=counts.get(k,0)+v
            del captures,z,loss
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        gradients=[float(obj.core.content.weight.grad.norm()) for obj in model.objects]
        assert all(x>0 for x in gradients)
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        rows.append(dict(arm=arm,parameters=sum(p.numel() for p in model.parameters()),paired_update_s=time.monotonic()-start,sensitivity=sens,content_gradients=gradients,work=counts,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    files=['experiments/hierarchy/carried_role_gate.py','sleeping_machines/carried_role_composition.py','experiments/theory/CARRIED_ROLE_REPRESENTATIONS.md']
    result=dict(status='completed',battle='G-HIER',stage='carried_role_gate',rows=rows,input_sha256={parent:sha(ROOT/parent)},source_sha256={f:sha(ROOT/f) for f in files},scope='One TRAIN pair/common noise,actual paired AdamW update per arm. Frozen acquired starting native/composer parameters,carrier gain0.5. Sensitivity/resource gate only;no generalization or efficiency claim. Cumulative process peak is not isolated memory comparison.')
    with (ROOT/f'experiments/results/hierarchy/{a.tag}.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
