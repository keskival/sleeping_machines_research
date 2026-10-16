"""Measured support repair: complete anonymous observed parts in native memory."""
import argparse,json,resource,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch
from torch.nn import functional as F
from sleeping_machines.observed_ownership_composition import ObservedOwnershipComposition
from experiments.hierarchy.grounded_composition_learning import scene_forward,PARENT
from experiments.hierarchy.part_combination_task import corpus
from experiments.hierarchy.native_composition_data import sha,source_hashes
AUDIT='experiments/results/hierarchy/aws_g_hier_ownership_support_20261016_v1.json'
SOURCES=['experiments/hierarchy/owned_native_preflight.py','sleeping_machines/observed_ownership_composition.py','sleeping_machines/bound_native_composition.py','experiments/hierarchy/grounded_composition_learning.py','experiments/hierarchy/part_combination_task.py','tests/test_observed_ownership_composition.py']
def main():
 p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args();torch.set_num_threads(1);start=time.monotonic()
 audit=json.loads((ROOT/AUDIT).read_text());assert audit['status']=='completed'
 row=next(r for r in corpus() if r.split=='train' and r.nuisance==0)
 torch.manual_seed(1301);model=ObservedOwnershipComposition(classes=2,ownership_threshold=audit['threshold'])
 model.load_state_dict(torch.load(PARENT,map_location='cpu',weights_only=True))
 torch.manual_seed(1303);z,state,_=scene_forward(model,row)
 destinations={a for a,_ in state.ownership.values()};edges=[]
 for dest in destinations:
  actual={p.source[1] for p in model.parts(state,1,dest) if p.source[0]==0}
  expected={p for p,(a,_) in state.ownership.items() if a==dest}
  assert actual==expected,'part is fragmented or mixed in its own native memory'
  edges.append(len(actual))
 F.cross_entropy(z[None],torch.tensor([row.label])).backward()
 assert all(o.core.content.weight.grad is not None and o.core.content.weight.grad.norm()>0 for o in model.objects)
 assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
 work=model.work(state);model.eval()
 with torch.no_grad():
  torch.manual_seed(1303);other,_,_=scene_forward(model,row)
 assert torch.equal(z.detach(),other),'native training/evaluation forward law changed'
 sources=source_hashes();sources.update({n:sha(ROOT/n) for n in SOURCES})
 r=dict(status='completed',battle='G-HIER',stage='owned_native_preflight',gate='PASS',levels=model.levels,native_stages=4*model.levels,parameters=sum(p.numel() for p in model.parameters()),ownership_threshold=audit['threshold'],complete_pure_part_memories=len(destinations),pixels_per_part=sorted(edges),observed_pixels=len(state.ownership),all_level_gradients_finite_and_nonzero=True,train_eval_bitwise_equal=True,work=work,process_wall_s=time.monotonic()-start,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,source_sha256=sources,input_sha256={AUDIT:sha(ROOT/AUDIT),str(PARENT.relative_to(ROOT)):sha(PARENT)},scope='One complete32event TRAIN scene;threshold fitted from anonymous TRAIN geometry. Existing acquired native weights,eight levels/32stages. Observed spatial groups have complete native memories before higher composition. Explicit static grouping,not learned semantic identity or general streaming ownership. CE backward only,no optimizer/acquisition,DEV class score or TEST.',decision='Retain the measured support repair if causality/native learning/resources pass;compare stable part identity and role interactions before admitting semantic acquisition.')
 with (ROOT/f'experiments/results/hierarchy/{a.tag}.json').open('x') as f:json.dump(r,f,indent=2);f.write('\n')
 print(json.dumps({k:r[k] for k in ['gate','complete_pure_part_memories','pixels_per_part','max_rss_kb','process_wall_s']}),flush=True)
if __name__=='__main__':main()
