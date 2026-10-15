"""Frozen completed grown actor: distinguish new-feature utility from base drift."""
import argparse,json,sys,time,types
from pathlib import Path
from types import SimpleNamespace
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'experiments'));sys.path.insert(0,str(Path(__file__).parent))
import protected_depth_local_gate as gate
from public_token_scoring import score

def main():
 p=argparse.ArgumentParser();p.add_argument('--tag',required=True);a=p.parse_args();torch.set_num_threads(1);began=time.monotonic();model,opt,args,train,warm,path=gate.restore_arm('grown');base=ROOT/'experiments/results/token_language/aws_protected_depth_grown_2m_20261014_s6_v1';r=json.loads(base.with_suffix('.json').read_text());assert r['status']=='completed';ck=torch.load(base.with_suffix('.pt'),weights_only=False);dev=gate.pilot.driver.load_tokens(args.dev_file);rows=[]
 for mode in ['trained_branch','zero_correction','parent_feature_substitution']:
  model.load_state_dict(ck['model']);model.core.new_depth_generator.manual_seed(args.seed+100001)
  if mode=='zero_correction':
   with torch.no_grad():model.readout.correction.weight.zero_()
  if mode=='parent_feature_substitution':model.readout.features=types.MethodType(lambda self,x:x[...,:self.parent_width]+self.correction(x[...,self.parent_width-model.core.total_payload:self.parent_width]),model.readout)
  elif 'features' in model.readout.__dict__:del model.readout.__dict__['features']
  result=score(model,dev,args,SimpleNamespace(Model=type(model)),25165824,8192,1024,8)
  if mode=='trained_branch':assert result==r['scored'],'Completed endpoint replay failed'
  rows.append(dict(mode=mode,scored=result));print(json.dumps(rows[-1]),flush=True)
 paths=[base.with_suffix('.json'),base.with_suffix('.pt')];sources=['experiments/credit/protected_depth_contribution_audit.py','experiments/credit/protected_depth_local_gate.py','sleeping_machines/protected_depth_language.py','sleeping_machines/protected_temporal_branch.py','experiments/public_token_scoring.py'];record=dict(status='completed',battle='R1',rows=rows,input_sha256={str(p.relative_to(ROOT)):gate.pilot.driver.sha(p) for p in paths},source_sha256={n:gate.pilot.driver.sha(ROOT/n) for n in sources},wall_s=time.monotonic()-began,scope='Frozen completed grown actor, identical8192DEVtargets/1024history/paired noise. Zero output correction isolates learned parent/readout drift within grown arm; parent-feature substitution tests distinct new features with same learned projection. No fitting, new public test, latency or heldout selection claim. Continuedparent comparison remains6.065562542; none of these ablations is a retrained model.')
 with (ROOT/f'experiments/results/diagnostics/{a.tag}.json').open('x') as f:json.dump(record,f,indent=2)
if __name__=='__main__':main()
