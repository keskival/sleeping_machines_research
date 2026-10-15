"""Source-bound G-HIER real-TRAIN resource/credit check. No optimizer or fit."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import torch
from sleeping_machines.addressable_parent_hierarchy import EventAddressableParentHierarchy
from experiments.hierarchy.visual_representation import recording


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True);args=parser.parse_args()
    torch.set_num_threads(1);torch.manual_seed(31)
    start=time.monotonic()
    directory=ROOT/'data/dvsgesture/DvsGesture'
    train=directory/'trials_to_train.txt'
    names=sorted(n.strip() for n in train.read_text().splitlines() if n.endswith('.aedat') and 1<=int(n[4:6])<=16)
    path=directory/names[0]
    counts,labels=recording(path)
    events=torch.tensor(counts[:1],dtype=torch.float32)
    target=torch.tensor([labels[0][0]])
    model=EventAddressableParentHierarchy(width=32,levels=8,refinements=4,roles=4)
    # Uniform cell-balanced audit, independent of the race tape.
    site,q=model.sample_audit_site(events,torch.Generator().manual_seed(37))
    loss,info=model.route_objective(events,target,site,seed=41,site_probability=q)
    loss.backward()
    gradients={name:float(p.grad.abs().sum()) for name,p in model.named_parameters() if p.grad is not None}
    if not torch.isfinite(loss) or any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
        raise ValueError('nonfinite loss or gradient')
    output=info.pop('output')
    with torch.no_grad():
        continued=model(None,output['memory'],seed=43)
    files=['sleeping_machines/addressable_parent_hierarchy.py','sleeping_machines/parent_race_hierarchy.py',
        'sleeping_machines/spatial_race_hierarchy.py','experiments/hierarchy/visual_representation.py',
        'experiments/hierarchy/addressable_preflight.py','tests/test_addressable_parent_hierarchy.py']
    paths=[train,path,path.with_name(path.stem+'_labels.csv')]
    result=dict(status='completed',battle='G-HIER',stage='addressable_preflight',
        scope='one official TRAIN recording, first gesture; no optimizer, no fitting, no test data; not a quality score',
        source_sha256={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in files},
        input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        input_shape=list(events.shape),parameters=sum(p.numel() for p in model.parameters()),
        process_wall_s=time.monotonic()-start,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        gradient_l1=gradients,factual_loss=float(info['factual']),audit_probability=q,replays=info['replays'],
        memory_continuation_finite=bool(torch.isfinite(continued['logits']).all()),
        resource_accounting=model.work_descriptor(128,128),
        next_decision='Signal/gradient variance and bounded fit admission; no automatic long fit from this single example')
    out=ROOT/f'experiments/results/hierarchy/{args.tag}.json';out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['gradient_l1','resource_accounting','source_sha256','input_sha256']}))


if __name__=='__main__': main()
