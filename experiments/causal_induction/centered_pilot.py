"""Diagnosed native credit teacher repair; same completed actor/audit controls."""
import argparse,hashlib,json,resource,time
from pathlib import Path
import numpy as np
import torch
from native_sensor import ROOT
from credit_run import run
from centered_race_credit import CenteredRaceCredit
from public_stream import persistence
class ResetCentered(CenteredRaceCredit):
 def issue(self,r,st,generator,arm='credit',learning=False):
  st['credit_nodes']={};return super().issue(r,st,generator,arm,learning)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);a=ap.parse_args();torch.set_num_threads(1);out=ROOT/f'experiments/results/causal_induction/{a.tag}.json';assert not out.exists();begin=time.monotonic();parent=ROOT/'experiments/results/causal_induction/curie_causal_induction_20261012_v1_prepare.json';data=json.loads(parent.read_text());contract=ROOT/'experiments/results/causal_induction/curie_causal_centered_credit_20261012_v3_contract.json';check=json.loads(contract.read_text());assert check['status']=='completed';rows=[]
 def path(x):
  p=ROOT/x['path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==x['sha256'];return p
 train=[x for x in data['records'] if x['split']=='train'][:8];dev=[x for x in data['records'] if x['split']=='heldout_graph' and x['variant']=='base'][:4];prior=(1.,2.);start=time.monotonic()
 for x in train:_,prior=persistence(path(x),prior,True)
 priorwall=time.monotonic()-start;confidence=prior[0]/prior[1];torch.manual_seed(194);initial=CenteredRaceCredit(confidence).state_dict()
 for label,cls in [('centered_credit',CenteredRaceCredit),('centered_reset',ResetCentered)]:
  m=cls(confidence);m.load_state_dict(initial);opt=torch.optim.Adam(m.parameters(),lr=.001);acq=[]
  for epoch in range(2):
   for i,x in enumerate(train):acq.append(dict(epoch=epoch,**run(m,path(x),'credit',opt,seed=194+epoch*100+i)))
  final=[run(m,path(x),'credit',seed=900+i) for i,x in enumerate(dev)];cp=out.with_name(out.stem+'_'+label+'.pt');torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict(),confidence=confidence),cp);rows.append(dict(arm=label,train=acq,dev=final,mean_dev_nll=float(np.mean([x['nll'] for x in final])),whole_fit_wall_s=priorwall+sum(x['wall_s'] for x in acq),checkpoint=str(cp.relative_to(ROOT)),checkpoint_sha256=hashlib.sha256(cp.read_bytes()).hexdigest()))
 sources=[Path(__file__),ROOT/'experiments/causal_induction/centered_race_credit.py',ROOT/'experiments/causal_induction/race_credit_sensor.py',ROOT/'experiments/causal_induction/structural_sensor.py',ROOT/'experiments/causal_induction/native_sensor.py',ROOT/'experiments/causal_induction/credit_run.py',ROOT/'experiments/causal_induction/public_stream.py',ROOT/'sleeping_machines/addressed_event_heads.py',ROOT/'sleeping_machines/paired_route_credit.py',ROOT/'sleeping_machines/parallel_head_race_language.py',ROOT/'sleeping_machines/sparse_race_language.py'];out.write_text(json.dumps(dict(status='completed',battle='B5/R1',tag=a.tag,rows=rows,prior_acquisition_targets=512,whole_pipeline_wall_s=time.monotonic()-begin,peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),contract_sha256=hashlib.sha256(contract.read_bytes()).hexdigest(),source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},scope='Diagnosed native temporal critic teacher: route loss minus same issued persistence baseline. Forecast actor/credit architecture,194initialization,8TRAIN graphs/two passes/prior acquisition/audit and noise schedules unchanged. Saved exact/zero25 controls retained; centered teacher exact-actor parity contract0.0. Original four selection graphs, not a confirmation. No dense controller, hidden labels, publicTEST or efficiency claim.'),indent=2)+'\n');print(json.dumps(dict(rows=[{k:x[k] for k in ['arm','mean_dev_nll','whole_fit_wall_s']} for x in rows])),flush=True)
if __name__=='__main__':main()
