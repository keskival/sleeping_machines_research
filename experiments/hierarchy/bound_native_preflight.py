"""G-HIER: acquired eight-level native actor,part-version and race-pool gate."""
import argparse,json,resource,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch
from torch.nn import functional as F
from sleeping_machines.relational_frame_composition import RelationalFrameComposition
from sleeping_machines.bound_native_composition import BoundNativeComposition
from sleeping_machines.bound_recursive_race_learning import BoundRecursiveRaceHeads
from experiments.hierarchy.grounded_composition_learning import scene_forward,PARENT
from experiments.hierarchy.part_combination_task import corpus
from experiments.hierarchy.native_composition_data import sha,source_hashes
SOURCES=['experiments/hierarchy/bound_native_preflight.py','sleeping_machines/bound_native_composition.py','sleeping_machines/bound_recursive_race_learning.py','sleeping_machines/recursive_race_learning.py','sleeping_machines/race_readout.py','sleeping_machines/relational_frame_composition.py','experiments/hierarchy/grounded_composition_learning.py','experiments/hierarchy/part_combination_task.py','tests/test_bound_native_composition.py','tests/test_bound_recursive_race_learning.py']

def main():
 p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args();torch.set_num_threads(1);began=time.monotonic()
 row=next(r for r in corpus() if r.split=='train' and r.nuisance==0)
 weights=torch.load(PARENT,map_location='cpu',weights_only=True);measurements=[];grads=None;baseline=None
 for name,cls in [('acquired',RelationalFrameComposition),('bound',BoundNativeComposition)]:
  torch.manual_seed(1213);model=cls(classes=2);model.load_state_dict(weights)
  torch.manual_seed(1223);start=time.monotonic();z,state,_=scene_forward(model,row)
  F.cross_entropy(z[None],torch.tensor([row.label])).backward(retain_graph=name=='bound')
  current={n:None if p.grad is None else p.grad.detach().clone() for n,p in model.named_parameters()}
  if baseline is None:baseline=z.detach().clone();ready=state.now;grads=current
  else:
   assert torch.equal(z.detach(),baseline) and state.now==ready,'forward schedule changed'
   assert all((grads[n] is None and g is None) or (grads[n] is not None and g is not None and torch.equal(grads[n],g)) for n,g in current.items()),'native parameter gradient changed'
  measurements.append(dict(arm=name,forward_backward_s=time.monotonic()-start,work=model.work(state),logits=z.detach().tolist(),ready=state.now))
  if name=='acquired':del model,state,z
 # Real eight-level bindings,then one local native race prediction from root.
 parts=model.parts(state,7,(0,0));assert len(parts)>1,'actual root does not pool multiple child memories'
 head=BoundRecursiveRaceHeads(model.width,model.levels,slots=len(parts),identities=8,hidden=32)
 model.zero_grad(set_to_none=True);head.zero_grad(set_to_none=True)
 ledger,work=head.forecast_parts(model,state,7,(0,0))
 # Contract target only: an arbitrary legal future identity checks learning
 # connectivity. No fit/semantic/likelihood quality claim uses this label.
 local=ledger.silence(ledger.cursor+1.)+ledger.event(ledger.cursor+2.,torch.tensor([0]))[0]
 local.sum().backward()
 assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
 assert head.heads[7].out.weight.grad.norm()>0
 assert model.objects[6].core.content.weight.grad.norm()>0
 before=len(state.bindings.entries);removed=model.collect_bindings(state)
 for part in model.parts(state,7,(0,0)):assert part.source in state.bindings.entries
 sources=source_hashes();sources.update({n:sha(ROOT/n) for n in SOURCES})
 result=dict(status='completed',battle='G-HIER',stage='bound_native_preflight',gate='PASS',levels=model.levels,native_stages=model.levels*4,
  parameters=sum(p.numel() for p in model.parameters()),bitwise_forward_schedule_and_gradient_parity=True,
  measurements=measurements,bound_part_pool=len(parts),local_race_work=work,root_race_head_parameters=sum(p.numel() for p in head.parameters()),
  retained_records_before_gc=before,garbage_collected_records=removed,final_work=model.work(state),local_gradients_finite=True,
  input_sha256={str(PARENT.relative_to(ROOT)):sha(PARENT)},source_sha256=sources,
  process_wall_s=time.monotonic()-began,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
  scope='Acquired original relational actor,8abstraction levels/32native stages,one complete32event TRAIN scene. No acquisition or TEST access. Baseline CE gradients are checked bitwise. Arbitrary legal future mark tests only local race connectivity,not prediction quality or semantic identities. Extra retained versions and metadata are charged;not a full FLOP/hardware result.',
  decision='Retain versioned part access if parity/causality/resources pass;next semantic admission must supply stable observable identities and relation-relevance tests. No automatic larger fit.')
 with (ROOT/f'experiments/results/hierarchy/{a.tag}.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps({k:result[k] for k in ['gate','bound_part_pool','parameters','max_rss_kb','process_wall_s']}),flush=True)
if __name__=='__main__':main()
