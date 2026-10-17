"""Admit source-bound calibrated reciprocal gates and prospective continuation."""
import hashlib,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/'experiments/queue/aws_model_improvement_repair_20261005T161000Z'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 assert os.uname().nodename=='ip-172-31-47-132'
 old=json.loads((PLAN/'addenda/000_aws_r1_00000_reciprocal_future_20261016_v6.json').read_text())
 declaration=dict(old['purpose'])
 declaration.update(fighter='Best saved ordinary eight-layer native actor with acquired native credit core, neutral action-scaled recursive least-squares response and independent audit policy.',restrictions='Single-parent chronological TRAIN development. Service action chosen before outcomes with uncertainty penalty; separate uniform full-support nonzero-action audits. Two gate episodes then 64 prospective episodes, four support and 32 future targets each. No TEST, rollback, whole-history exact metagradient or hardware claim.',accounting='Count every actual, audit, control and both native-reference trajectory, critic update, snapshot and replay. One existing gym slot, one thread, 3M KiB RSS/12M KiB VMS, at least 8 GiB MemAvailable. Atomic joint episode recovery. No full arithmetic or matched-work win claim.',decision='Does calibrated future-value learning improve subsequent prediction over ordinary and immediate-greedy native learning? Gate validates neutrality, causal finite credit and recovery first; report action calibration independently.')
 sources=set(old['slots']['3'][0]['source_sha256'])
 sources.update(str(p.relative_to(ROOT)) for p in (ROOT/'sleeping_machines').glob('*.py'))
 sources.update(['experiments/credit/r1_calibrated_reciprocal.py','tests/test_calibrated_native_future_credit.py'])
 hashes={n:sha(ROOT/n) for n in sorted(sources)}
 jobs=[];parent='experiments/results/diagnostics/aws_r1_native_reciprocal_online_1152_20261016_v6.json'
 stages=['calibrated native reciprocal numerical gate','calibrated native reciprocal acquired gate']
 for i,mode in enumerate(['contract','acquired','continue']):
  tag=f'aws_r1_calibrated_reciprocal_{mode}_20261017_v1'
  q=ROOT/f'experiments/queue/r1/{tag}/{tag}.txt';q.parent.mkdir(parents=True,exist_ok=True)
  text='# '+json.dumps(declaration)+'\n'+f'{tag} experiments/credit/r1_calibrated_reciprocal.py --mode {mode} --tag {tag}'+('' if i==0 else ' --parent '+parent)+'\n'
  if q.exists():assert q.read_text()==text
  else:q.write_text(text)
  result=f'experiments/results/diagnostics/{tag}.json'
  jobs.append(dict(name=tag,kind='r1',queue=str(q.relative_to(ROOT)),queue_sha256=sha(q),source_sha256=hashes,result=result,requires=[] if i==0 else [dict(result=jobs[-1]['result'],stage=stages[i-1])],timeout_s=[1800,7200,21600][i],rss_kb=3000000,vms_kb=12000000))
  if i==1:parent=result
 packet=dict(host=os.uname().nodename,battle='R1',purpose=declaration,parent_manifest_sha256=sha(PLAN/'manifest.json'),slots={'3':jobs})
 path=PLAN/'addenda/000_aws_r1_00000_calibrated_reciprocal_20261017_v1.json'
 if path.exists():assert json.loads(path.read_text())==packet
 else:
  tmp=path.with_suffix('.writing');tmp.write_text(json.dumps(packet,indent=2)+'\n');tmp.replace(path)
 print(path)
if __name__=='__main__':main()
