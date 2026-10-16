"""Replace unstarted RMR2 admissions with a causally equivalent stable implementation."""
import hashlib,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];PLAN=ROOT/'experiments/queue/aws_model_improvement_repair_20261005T161000Z'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 assert os.uname().nodename=='ip-172-31-47-132'
 sources=['experiments/hierarchy/rmr2_stable_decay_gate.py','experiments/r1_rmr2_stable.py','experiments/r1_rmr2_stable_stream.py','experiments/r1_rmr2.py','experiments/r1_rmr.py','experiments/r1_cerl.py','experiments/r1_deep_keyed_lm.py','experiments/r1_token_keyed_lm.py'];slots={}
 for slot,act in [(3,'act0'),(5,'act01')]:
  old=f'aws_r1_rmr2_{act}_K4_4M_s0_20261016_v1';tag=old[:-2]+'v2';gate=tag+'_gate'
  for t in [old,old+'_gate']:
   q=ROOT/f'experiments/queue/r1/{t}/{t}.txt'
   assert not (ROOT/f'experiments/queue/logs/{t}.log').exists(),'Cannot withdraw a started source-bound job'
   if q.exists():q.rename(q.with_suffix('.withdrawn'))
  previous=ROOT/f'experiments/queue/r1/{old}/{old}.withdrawn'
  declaration='''Battle R1, existing PRODUCT_ORDERS firing/standing-memory comparison. Repair arithmetic, not architecture: truncate impossible future standing writes before exponentiation so factors grow by at most one bounded chunk. Same weights, laws, lower carrier, learning recipe and fixed-weight development protocols. Source-bound active jobs untouched. This model uses temporal-memory carrier layers, BPTT, lane-bounded state and soft expected firing during training; not a fully native event-local flagship or sparse-dispatch measurement. All discovery/backward/optimizer/traffic charged. Numerical long-window8192 oracle/finite-gradient gate before acquisition; no extra data, TEST or dense baseline. Original RMR2 v1 admissions withdrawn before execution. Complete structural/regularization arguments now included in checkpoint identity, including standing count and activity cost; checkpoint also binds sources. One thread,3MKiB RSS/12MKiB VMS watchdogs,8GiB reserve. Decision: can useful standing-memory refinements and learned firing improve the completed causal race language result?'''
  jobs=[]
  for t,script,args,res,req in [(gate,'experiments/hierarchy/rmr2_stable_decay_gate.py',f'--tag {gate}',f'experiments/results/diagnostics/{gate}.json',[]),(tag,'experiments/r1_rmr2_stable_stream.py',f'--tag {tag} --levels 4 --top-m 4 --act-cost {0 if act=="act0" else .01} --train-tokens 4194304 --seed 0',f'experiments/results/token_language/{tag}.json',[dict(result=f'experiments/results/diagnostics/{gate}.json',stage='RMR2 stable decay admission')])]:
   q=ROOT/f'experiments/queue/r1/{t}/{t}.txt';q.parent.mkdir(parents=True,exist_ok=True);text='# '+declaration+'\n'+t+' '+script+' '+args+'\n'
   if q.exists():assert q.read_text()==text
   else:q.write_text(text)
   jobs.append(dict(name=t,kind='r1',queue=str(q.relative_to(ROOT)),queue_sha256=sha(q),source_sha256={s:sha(ROOT/s) for s in sources},result=res,requires=req,timeout_s=1800 if t==gate else 100000,rss_kb=3000000,vms_kb=12000000))
  slots[str(slot)]=jobs
 packet=dict(host=os.uname().nodename,battle='R1',purpose=declaration,parent_manifest_sha256=sha(PLAN/'manifest.json'),slots=slots)
 out=PLAN/'addenda/000_aws_r1_rmr2_stable_decay_20261016_v1.json'
 if out.exists():assert json.loads(out.read_text())==packet
 else:
  tmp=out.with_suffix('.writing');tmp.write_text(json.dumps(packet,indent=2)+'\n');tmp.replace(out)
 print('Admitted replacement gates and conditional fits; no job starts before its long-window gate passes.')
if __name__=='__main__':main()
