"""G-HIER real TRAIN input contract for a new local binding teacher; no fit."""
import argparse, json, resource, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch
from experiments.hierarchy.composition_credit_audit import CHECKPOINTS,AUDIT
from experiments.hierarchy.grounded_composition_learning import scene_forward
from experiments.hierarchy.part_combination_task import corpus
from experiments.hierarchy.native_composition_data import sha,source_hashes
from sleeping_machines.part_identity_composition import fit_shape_vocabulary
from sleeping_machines.identity_role_composition import IdentityRoleComposition
from sleeping_machines.binding_race_teacher import BindingRaceTeacher,observed_binding_race_loss


def main():
 p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args()
 torch.set_num_threads(1);began=time.monotonic();rows=[r for r in corpus() if r.split=='train' and r.nuisance==0]
 threshold=json.loads((ROOT/AUDIT).read_text())['threshold'];vocab=fit_shape_vocabulary([r.events for r in rows],threshold)
 torch.manual_seed(1451);actor=IdentityRoleComposition(classes=2,ownership_threshold=threshold,shape_vocabulary=vocab)
 actor.load_state_dict(torch.load(ROOT/CHECKPOINTS[512],map_location='cpu',weights_only=False)['model']);actor.train()
 torch.manual_seed(161016);teacher=BindingRaceTeacher(actor.width)
 torch.manual_seed(113);logits,state,_=scene_forward(actor,rows[0]);before=logits.detach().clone()
 loss,work=observed_binding_race_loss(actor,state,teacher);loss.backward()
 norms={}
 for name,param in actor.named_parameters():
  if param.grad is not None:
   assert bool(torch.isfinite(param.grad).all())
   block='level_'+name.split('.')[1] if name.startswith('objects.') else name.split('.')[0]
   norms[block]=norms.get(block,0.)+float(param.grad.square().sum())
 assert bool(torch.isfinite(loss)) and any(v>0 for v in norms.values())
 assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in teacher.parameters())
 native_work=actor.work(state);del state,logits
 torch.manual_seed(113);again,state,_=scene_forward(actor,rows[0]);parity=torch.equal(before,again.detach());assert parity
 sources=source_hashes()
 for f in ['sleeping_machines/binding_race_teacher.py','experiments/hierarchy/binding_race_gate.py','tests/test_binding_race_teacher.py']:sources[f]=sha(ROOT/f)
 result=dict(status='completed',battle='G-HIER',stage='binding_race_gate',gate='PASS',loss=float(loss.detach()),work=work,native_work=native_work,actor_gradient_norms={k:v**.5 for k,v in norms.items()},actor_output_unchanged=parity,teacher_parameters=sum(p.numel() for p in teacher.parameters()),wall_s=time.monotonic()-began,max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,source_sha256=sources,input_sha256={AUDIT:sha(ROOT/AUDIT),CHECKPOINTS[512]:sha(ROOT/CHECKPOINTS[512])},scope='One original TRAIN scene and frozen512update actor;actual issued-version binding teacher backward,no optimizer step,DEV,TEST,ablation or predictive quality claim. Teacher-only gradient gate is not an acquisition.')
 target=ROOT/f'experiments/results/hierarchy/{a.tag}.json';assert not target.exists();target.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['gate','loss','work','wall_s','max_rss_kb']}),flush=True)
if __name__=='__main__':main()
