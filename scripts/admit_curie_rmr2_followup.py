"""Admit new RMR2 orders; rebind only unstarted CERL queues after metadata-only import change."""
import hashlib,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];PLAN=ROOT/'experiments/queue/aws_model_improvement_repair_20261005T161000Z'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def job(tag,q,src,result,requires=[]):return dict(name=tag,kind='r1',queue=str(q.relative_to(ROOT)),queue_sha256=sha(q),source_sha256={s:sha(ROOT/s) for s in src},result=result,requires=requires,timeout_s=100000,rss_kb=3000000,vms_kb=12000000)
def main():
 assert os.uname().nodename=='ip-172-31-47-132';slots={}
 base=['experiments/r1_cerl.py','experiments/r1_deep_keyed_lm.py','experiments/r1_token_keyed_lm.py']
 # Preserve immutable v1 queue text under withdrawn suffix; changed source binding gets a unique admission/tag.
 for slot,suffix in [(2,'cerl_pk262144_4M'),(6,'cerl_msg_4M')]:
  old=f'aws_r1_{suffix}_s0_20261016_v1';tag=old[:-2]+'v2';oq=ROOT/f'experiments/queue/r1/{old}/{old}.txt';withdrawn=oq.with_suffix('.withdrawn')
  if oq.exists():
   assert not (ROOT/f'experiments/queue/logs/{old}.log').exists(),'job already started'
   oq.rename(withdrawn)
  text=withdrawn.read_text().replace(old,tag)+'# v1 unstarted admission withdrawn: unrelated RMR import metadata changed after pull; bind only actual CERL imports. No acquisition repeated.\n'
  q=ROOT/f'experiments/queue/r1/{tag}/{tag}.txt';q.parent.mkdir(parents=True,exist_ok=True)
  if q.exists():assert q.read_text()==text
  else:q.write_text(text)
  slots[str(slot)]=[job(tag,q,base,f'experiments/results/token_language/{tag}.json')]
 for slot,act in [(3,'act0'),(5,'act01')]:
  tag=f'aws_r1_rmr2_{act}_K4_4M_s0_20261016_v1';gate=tag+'_gate';src=base+['experiments/r1_rmr.py','experiments/r1_rmr2.py','experiments/hierarchy/rmr2_admission_gate.py'];q=ROOT/f'experiments/queue/r1/{gate}/{gate}.txt';q.parent.mkdir(parents=True,exist_ok=True)
  text=f'# Battle R1: current RMR2 causality, normalization, zero-refinement and hard-fire numerical admission; no fit.\n{gate} experiments/hierarchy/rmr2_admission_gate.py --tag {gate}\n'
  if q.exists():assert q.read_text()==text
  else:q.write_text(text)
  slots[str(slot)]=[job(gate,q,src,f'experiments/results/diagnostics/{gate}.json'),job(tag,ROOT/f'experiments/queue/r1/{tag}/{tag}.txt',src,f'experiments/results/token_language/{tag}.json',[dict(result=f'experiments/results/diagnostics/{gate}.json',stage='RMR2 numerical admission')])]
 packet=dict(host=os.uname().nodename,battle='R1',purpose='Latest PRODUCT_ORDERS: RMR2 firing and standing slots after current fits; fresh same-protocol4Mdevelopment, temporal carrier/BPTT and lane reset explicitly retained. Soft expected writes in training; hard firing does not imply sparse dispatch. All discovery, correction, learning and storage charged; one-thread3MKiB RSS/12MKiB VMS watchdogs. Numerical gate precedes each RMR2fit. Preserve current source-bound loaded code and results; no TEST or dense training.',parent_manifest_sha256=sha(PLAN/'manifest.json'),slots=slots)
 out=PLAN/'addenda/000_aws_r1_curie_rmr2_followup_20261016_v1.json'
 if out.exists():assert json.loads(out.read_text())==packet
 else:
  tmp=out.with_suffix('.writing');tmp.write_text(json.dumps(packet,indent=2)+'\n');tmp.replace(out)
 print(out)
if __name__=='__main__':main()
