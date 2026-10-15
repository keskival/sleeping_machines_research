"""Frozen native audit: separate translation from timing/order nuisance changes."""
import argparse,json,resource,sys,time
from dataclasses import replace
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from experiments.hierarchy.part_combination_task import render
from experiments.hierarchy.coalesced_frozen_readout import features
from experiments.hierarchy.native_composition_data import source_hashes,sha
from sleeping_machines.coalesced_native_composition import CoalescedNativeComposition


def factorial(left,right,root):
    base=render(left,right,root,0);both=render(left,right,root,2)
    translation=lambda n:np.random.default_rng(1771+n).integers(-3,4,size=2)
    displacement=translation(2)-translation(0)
    translated=base.events.copy();translated[:,1:3]+=displacement
    mapping={(int(y-displacement[0]),int(x-displacement[1]),int(p)):int(t) for t,y,x,p in both.events}
    timing=base.events.copy()
    for row in timing:row[0]=mapping[tuple(int(v) for v in row[1:])]
    timing=timing[np.argsort(timing[:,0],kind='stable')]
    return [base,replace(base,events=translated),replace(base,events=timing),both]


def main():
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args()
    started=time.monotonic();torch.set_num_threads(1)
    rows=[]
    for left,difference in enumerate((0,1,3,7)):
        for root in (0,1):rows+=factorial(left,left^difference,root)
    torch.manual_seed(211);initial=CoalescedNativeComposition(classes=2)
    old=ROOT/'experiments/results/hierarchy/aws_g_hier_native_composition_20261015_v1_learn.final.pt'
    initial.load_state_dict({k:v for k,v in torch.load(old,map_location='cpu',weights_only=True).items() if not k.startswith('head.')},strict=False)
    final=CoalescedNativeComposition(classes=2)
    final_path=ROOT/'experiments/results/hierarchy/aws_g_hier_coalesced_learning_20261015_v1.final.pt'
    final.load_state_dict(torch.load(final_path,map_location='cpu',weights_only=True))
    records=[];packet={}
    for name,model in [('initial',initial),('final',final)]:
        x,y,work=features(model,rows);array=x.numpy().reshape(4,2,4,-1)
        deltas={label:float(np.linalg.norm(array[:,:,i]-array[:,:,0],axis=-1).mean())
                for i,label in [(1,'translation_only'),(2,'timing_only'),(3,'both')]}
        relation=float(np.linalg.norm(array[:,0,0]-array[:,1,0],axis=-1).mean())
        records.append(dict(variant=name,feature_displacement=deltas,root_relation_displacement=relation,work=work))
        packet[name]=array
    out=ROOT/f'experiments/results/hierarchy/{a.tag}.json';cache=out.with_name(out.stem+'_features.npz')
    np.savez_compressed(cache,**packet)
    sources=source_hashes()
    for f in ['experiments/hierarchy/coalesced_nuisance_factorial.py','experiments/hierarchy/coalesced_frozen_readout.py','experiments/hierarchy/part_combination_task.py']:sources[f]=sha(ROOT/f)
    result=dict(status='completed',battle='G-HIER',stage='coalesced_nuisance_factorial',rows=records,
      source_sha256=sources,input_sha256={str(p.relative_to(ROOT)):sha(p) for p in [old,final_path]},
      feature_packet=dict(path=str(cache.relative_to(ROOT)),sha256=sha(cache)),process_wall_s=time.monotonic()-started,
      max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
      scope='Four TRAIN child pairs,two root orientations,translation/timing2x2,initial/final frozen cores,common tape113. No label fitting or DEV/TEST model scoring. Single tape includes stochastic scheduling sensitivity.',
      decision='Localize the measured nuisance domination before choosing role/ownership versus temporal-order repair; no unchanged fitting follows.')
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(rows=records,wall_s=result['process_wall_s'])),flush=True)
if __name__=='__main__':main()
