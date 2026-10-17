"""Recover acquired-native composition evidence without refitting its finished arm."""
import hashlib,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];PLAN=ROOT/'experiments/queue/aws_model_improvement_repair_20261005T161000Z'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 assert os.uname().nodename=='ip-172-31-47-132'
 old=json.loads((PLAN/'addenda/000_aws_r1_001_composition_empirical_20261016_v2.json').read_text())
 purpose=dict(old['purpose']);purpose.update(state_history='Carry all numerical state and Adam across a declared TRAIN-to-DEV tape splice. Last revealed TRAIN target forecasts first DEV target. Score every DEV target exactly once, with causal batch32 adaptation on both native arms. Frozen canonical DEV diagnostic separately unchanged.',accounting='Reuse source-verified completed joint-arm 65536-target training and frozen evaluation. Finish its8192 online targets; fit the previously unstarted strongest candidate from the identical saved parent for65536 targets,then8192online targets. Charge inherited, recovered and new work separately in raw records; no external model fit. Slot2,one thread,3MKiB RSS/12MKiB VMS,8GiB reserve;36h maximum.',decision='Repair the actual next-input protocol failure, recover completed work and decide whether full-depth sampled actual-child credit improves useful native composition. This is the already admitted comparison with corrected tape transition, not a new sweep.')
 sources=set(old['slots']['2'][0]['source_sha256']);sources.update(['experiments/r1_composition_credit_recovery.py','experiments/credit/persistent_tape_transition.py','experiments/credit/persistent_transition_gate.py','tests/test_persistent_tape_transition.py'])
 hashes={n:sha(ROOT/n) for n in sorted(sources)}
 jobs=[];gate='aws_r1_persistent_transition_gate_20261017_v1';tag='aws_r1_composition_credit_recovery_64k_20261017_v1'
 for name,script,args,result,requires,timeout in [(gate,'experiments/credit/persistent_transition_gate.py',f'--tag {gate}',f'experiments/results/diagnostics/{gate}.json',[],1800),(tag,'experiments/r1_composition_credit_recovery.py',f'--tag {tag} --recover-tag aws_r1_composition_credit_pair_64k_s0_20261016_v2 --extra-targets 65536 --online-targets 8192',f'experiments/results/token_language/{tag}.json',[dict(result=f'experiments/results/diagnostics/{gate}.json',stage='persistent native tape transition gate')],129600)]:
  q=ROOT/f'experiments/queue/r1/{name}/{name}.txt';q.parent.mkdir(parents=True,exist_ok=True);text='# '+json.dumps(purpose)+'\n'+f'{name} {script} {args}\n'
  if q.exists():assert q.read_text()==text
  else:q.write_text(text)
  jobs.append(dict(name=name,kind='r1',queue=str(q.relative_to(ROOT)),queue_sha256=sha(q),source_sha256=hashes,result=result,requires=requires,timeout_s=timeout,rss_kb=3000000,vms_kb=12000000))
 packet=dict(host=os.uname().nodename,battle='R1',purpose=purpose,parent_manifest_sha256=sha(PLAN/'manifest.json'),slots={'2':jobs})
 path=PLAN/'addenda/000_aws_r1_composition_credit_recovery_20261017_v1.json'
 if path.exists():assert json.loads(path.read_text())==packet
 else:
  temp=path.with_suffix('.writing');temp.write_text(json.dumps(packet,indent=2)+'\n');temp.replace(path)
 print(path)
if __name__=='__main__':main()
