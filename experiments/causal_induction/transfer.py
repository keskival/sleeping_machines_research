"""Acquired native credit learner transfer, gated by unchanged confirmation."""
import argparse,copy,hashlib,json,resource,time
from pathlib import Path
import numpy as np
import torch
from native_sensor import ROOT
from race_credit_sensor import RaceCreditSensor
from race_credit_controls import ZeroAudit
from transfer_run import run

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);ap.add_argument('--mode',choices=['smoke','fit'],required=True);a=ap.parse_args();torch.set_num_threads(1);out=ROOT/f'experiments/results/causal_induction/{a.tag}.json';assert not out.exists();begin=time.monotonic();parent=ROOT/'experiments/results/causal_induction/curie_causal_nonlinear_20261012_v2_prepare.json';data=json.loads(parent.read_text());refpath=ROOT/'experiments/results/causal_induction/curie_causal_race_credit_20261012_v1_fit.json';ref=json.loads(refpath.read_text());credit=next(x for x in ref['rows'] if x['arm']=='credit');cp=ROOT/credit['checkpoint'];assert hashlib.sha256(cp.read_bytes()).hexdigest()==credit['checkpoint_sha256'];source=torch.load(cp,weights_only=False,map_location='cpu');cases=[]
 for split in ['heldout_graph','heldout_factors','heldout_depth','heldout_factors_depth']:
  base=[x for x in data['records'] if x['split']==split and x['variant']=='base'];rewire=[x for x in data['records'] if x['split']==split and x['variant']=='rewire'];cases.extend([base[0],rewire[1]])
 if a.mode=='smoke':cases=[cases[-1]]
 rows=[];methods=[('exact',RaceCreditSensor,'exact'),('zero25',ZeroAudit,'credit'),('acquired_credit',RaceCreditSensor,'credit'),('reset_credit',RaceCreditSensor,'credit')]
 if a.mode=='smoke':methods=[methods[0],methods[2]]
 for label,cls,arm in methods:
  measurements=[]
  for i,item in enumerate(cases):
   path=ROOT/item['path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256'];torch.manual_seed(194);m=cls(source['confidence']);fresh={k:v.detach().clone() for k,v in m.critic.state_dict().items()};m.load_state_dict(source['model']);opt=torch.optim.Adam(m.parameters(),lr=.001);opt.load_state_dict(copy.deepcopy(source['optimizer']))
   if label=='reset_credit':
    m.critic.load_state_dict(fresh)
    for p in m.critic.parameters():opt.state.pop(p,None)
   result=run(m,path,arm,opt,seed=777+i,max_queries=16 if a.mode=='smoke' else None);measurements.append(dict(split=item['split'],variant=item['variant'],depth=item['depth'],width=item['width'],path=item['path'],**result))
  count=sum(x['targets'] for x in measurements);targetwall=sum(x['wall_s'] for x in measurements);rows.append(dict(arm=label,episodes=measurements,targets=count,nll=sum(x['nll']*x['targets'] for x in measurements)/count,target_learning_wall_s=targetwall,source_acquisition_wall_s=credit['whole_fit_wall_s'],whole_learning_wall_s=credit['whole_fit_wall_s']+targetwall))
 sources=[Path(__file__),ROOT/'experiments/causal_induction/transfer_run.py',ROOT/'experiments/causal_induction/race_credit_controls.py',ROOT/'experiments/causal_induction/race_credit_sensor.py',ROOT/'experiments/causal_induction/structural_sensor.py',ROOT/'experiments/causal_induction/native_sensor.py',ROOT/'experiments/causal_induction/credit_run.py',ROOT/'experiments/causal_induction/public_stream.py',ROOT/'sleeping_machines/addressed_event_heads.py',ROOT/'sleeping_machines/paired_route_credit.py',ROOT/'sleeping_machines/parallel_head_race_language.py',ROOT/'sleeping_machines/sparse_race_language.py'];protocol=ROOT/'experiments/causal_induction/TRANSFER.md';out.write_text(json.dumps(dict(status='completed',battle='B5/R1',tag=a.tag,mode=a.mode,rows=rows,source_checkpoint_sha256=credit['checkpoint_sha256'],source_acquisition_target_presentations=1536,whole_pipeline_wall_s=time.monotonic()-begin,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),reference_sha256=hashlib.sha256(refpath.read_bytes()).hexdigest(),protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest(),source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},scope='Conditional native learner transfer; common acquired forward weights/moments, independently copied optimizer state. Public nonlinear unknown8/16-factor4/8-layer systems, base/hidden-rewire episodes. Prequential predictions issued before feedback; both actor and credit learning causal within episode. Reset only credit parameters/moments; numerical states fresh for each system. Generating-depth transfer, not neural-depth scaling. Source acquisition paid once per full arm. No publicTEST/Transformer/efficiency claim.'),indent=2)+'\n');print(json.dumps(dict(mode=a.mode,rows=[{k:x[k] for k in ['arm','nll','target_learning_wall_s','whole_learning_wall_s']} for x in rows])),flush=True)
if __name__=='__main__':main()
