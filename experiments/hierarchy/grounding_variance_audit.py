"""TRAIN-fitted constant target reference; no actor fitting or TEST access."""
import argparse,json,sys,time,resource
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch
from sleeping_machines.grounded_native_composition import ObservedGeometrySketch
from sleeping_machines.sparse_geometry_teacher import sparse_geometry_targets
from experiments.hierarchy.part_combination_task import corpus
from experiments.hierarchy.native_composition_data import sha


def constant_reference(training,validation):
    # Equal scene weight,then equal object weight within each scene; matches the
    # native auxiliary objective instead of overweighting busy scenes.
    center=torch.stack([x.mean(0) for x in training]).mean(0)
    loss=lambda rows:float(torch.stack([(x-center).square().mean() for x in rows]).mean())
    return dict(train_loss=loss(training),dev_loss=loss(validation)),center


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args();torch.set_num_threads(1);start=time.monotonic()
    teacher=ObservedGeometrySketch().double();banks={split:[[] for _ in range(8)] for split in ['train','dev']};root=[];labels=[];work={}
    for scene in corpus():
        if scene.split not in banks:continue
        events,label=scene.model_input();origin=tuple(events[:,1:3].min(0));targets,counts=sparse_geometry_targets(events,8,origin,teacher.directions)
        for level in range(8):banks[scene.split][level].append(torch.stack([v for (l,address),(v,n) in sorted(targets.items()) if l==level]))
        for k,v in counts.items():work[k]=work.get(k,0)+v
        if scene.split=='train':root.append(targets[(7,(0,0))][0]);labels.append(label)
    references=[constant_reference(tr,dv)[0] for tr,dv in zip(banks['train'],banks['dev'])]
    x=torch.stack(root);y=torch.tensor(labels);distance=torch.cdist(x,x);opposite=y[:,None]!=y[None,:]
    margin=float(distance[opposite].min());assert margin>1e-8
    frozen_path='experiments/results/hierarchy/aws_g_hier_grounded_decoder_only_20261016_v1.json'
    fit=json.loads((ROOT/frozen_path).read_text());assert fit['status']=='completed'
    frozen=fit['final']['grounding_dev']
    result=dict(status='completed',battle='G-HIER',stage='grounding_variance_audit',constant_by_level=references,frozen_encoder_dev_loss=frozen,frozen_encoder_skill=[1-v/r['dev_loss'] if r['dev_loss']>1e-14 else None for v,r in zip(frozen,references)],train_opposite_label_root_sketch_min_distance=margin,work=work,process_wall_s=time.monotonic()-start,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,input_sha256={frozen_path:sha(ROOT/frozen_path)},source_sha256={f:sha(ROOT/f) for f in ['experiments/hierarchy/grounding_variance_audit.py','sleeping_machines/sparse_geometry_teacher.py','sleeping_machines/grounded_native_composition.py']},scope='TRAIN-derived per-level constants,scene/object weighting matches auxiliary loss,DEV statistics only,no TEST,no actor fitting. Positive finite TRAIN separation is not universal information sufficiency. Reconstruction skill is not semantic transfer.')
    with (ROOT/f'experiments/results/hierarchy/{a.tag}.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
