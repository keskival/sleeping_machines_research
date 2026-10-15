"""Measure local observable-geometry credit without changing native inference."""
import argparse,json,resource,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch
from torch.nn import functional as F
from sleeping_machines.grounded_native_composition import GroundedNativeComposition
from experiments.hierarchy.part_combination_task import corpus
from experiments.hierarchy.observed_training_pairs import training_pairs
from experiments.hierarchy.native_composition_data import sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args();torch.set_num_threads(1)
    parent='experiments/results/hierarchy/aws_g_hier_carried_continued_20261015_v1.raw.pt'
    torch.manual_seed(941);model=GroundedNativeComposition(classes=2);model.load_relational_state(torch.load(ROOT/parent,map_location='cpu',weights_only=True));model.train()
    train=[r for r in corpus() if r.split=='train'];pair=training_pairs([r.model_input() for r in train])[0]
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01);optimizer.zero_grad(set_to_none=True)
    start=time.monotonic();samples=[];work={};parameters=[obj.core.content.weight for obj in model.objects]
    for index in pair:
        torch.manual_seed(113);events,label=train[index].model_input();state=model.new_state()
        for micros,y,x,pol in events:model.observe(state,(int(y),int(x)),int(micros)/1e6,[float(pol),1.])
        logits,_=model.compose(state);ce=F.cross_entropy(logits[None],torch.tensor([label]))
        seconds=events.copy().astype('float64');seconds[:,0]/=1e6
        local,record=model.grounding_loss(state,seconds)
        gc=torch.autograd.grad(ce,parameters,retain_graph=True);gg=torch.autograd.grad(local,parameters,retain_graph=True)
        samples.append(dict(label_loss=float(ce.detach()),local_loss=float(local.detach()),objects_per_level=record['objects_per_level'],label_gradient_norms=[float(g.norm()) for g in gc],local_gradient_norms=[float(g.norm()) for g in gg],gradient_cosines=[float((x*y).sum()/(x.norm()*y.norm()+1e-30)) for x,y in zip(gc,gg)]))
        ((ce+.1*local)/2).backward()
        for k,v in model.work(state).items():
            if isinstance(v,int):work[k]=work.get(k,0)+v
        del logits,ce,local,record,state,gc,gg
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert all(g>0 for s in samples for g in s['local_gradient_norms'])
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
    files=['sleeping_machines/grounded_native_composition.py','experiments/hierarchy/grounding_gate.py','experiments/theory/LOCAL_OBSERVABLE_GROUNDING.md']
    result=dict(status='completed',battle='G-HIER',stage='grounding_gate',samples=samples,parameters=sum(p.numel() for p in model.parameters()),paired_update_s=time.monotonic()-start,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,work=work,input_sha256={parent:sha(ROOT/parent)},source_sha256={f:sha(ROOT/f) for f in files},scope='One observed TRAIN pair from acquired continued relational actor. Whole CE plus0.1fixed local geometry reconstruction. Two diagnostic autograd traversals plus actual backward per scene;no quality or training-efficiency claim. Native inference unchanged;teacher and decoders are training-only. No hidden IDs,DEV or TEST.')
    with (ROOT/f'experiments/results/hierarchy/{a.tag}.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
