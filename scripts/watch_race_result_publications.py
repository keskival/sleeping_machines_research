"""Recover completed race results independently of mutable-source post-fit checks.
Checks each result's recorded loaded-source hashes against its immutable admission.
Never rewrites a result, restarts a fit, or converts a pending score into evidence.
"""
import fcntl,hashlib,json,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];PLAN=ROOT/'experiments/queue/aws_model_improvement_repair_20261005T161000Z'
def main():
 while True:
  for packet in ['000_aws_r1_00000_calibrated_reciprocal_20261017_v1.json','000_aws_r1_00000_reciprocal_future_20261016_v5.json','000_aws_r1_00000_reciprocal_future_20261016_v4.json','000_aws_r1_00000_compact_recovery_20261016_v1.json','000_aws_r1_00001_compact_depth16M_20261016_v2.json','000_aws_r1_curie_ordered_program_20261016_v1.json','000_aws_r1_curie_rmr2_followup_20261016_v1.json','000_aws_r1_rmr2_stable_decay_20261016_v1.json','000_aws_r1_000_composition_learning_contract_20261016_v1.json','000_aws_r1_0000_credit_analytic_20261016_v1.json','000_aws_r1_0001_sampled_composition_gate_20261016_v1.json','000_aws_r1_001_composition_empirical_20261016_v1.json','000_aws_r1_0000_credit_analytic_20261016_v2.json','000_aws_r1_0001_sampled_composition_gate_20261016_v2.json','000_aws_r1_001_composition_empirical_20261016_v2.json']:
   admission=PLAN/'addenda'/packet
   if not admission.exists():continue
   for jobs in json.loads(admission.read_text())['slots'].values():
    for j in jobs:
     result=ROOT/j['result'];receipt=ROOT/f'.git/race-result-validation/{j["name"]}.json'
     if not result.exists() or receipt.exists():continue
     try:
      r=json.loads(result.read_text());assert r['status']=='completed'
      for name,digest in r['source_sha256'].items():assert j['source_sha256'][name]==digest,name
      row=dict(status='completed',battle='R1',stage='immutable-admission completed-result provenance check',run=j['name'],result=j['result'],result_sha256=hashlib.sha256(result.read_bytes()).hexdigest(),admission=str(admission.relative_to(ROOT)),admission_sha256=hashlib.sha256(admission.read_bytes()).hexdigest(),loaded_sources_match_admission=True,scope='Loaded source bindings match admission; current working source may differ after a pull. No predictive win inferred by this provenance check.')
      with open('/tmp/aws-language-publication.lock','a') as lock:
       fcntl.flock(lock,fcntl.LOCK_EX)
       if receipt.exists():continue
       receipt.parent.mkdir(parents=True,exist_ok=True);receipt.write_text(json.dumps(row,indent=2)+'\n')
       candidates=[result,admission,ROOT/j['queue'],ROOT/f'experiments/queue/logs/{j["name"]}.log']
       files=[str(p.relative_to(ROOT)) for p in candidates if p.exists()]
       subprocess.run(['git','add','-f','--',*files],cwd=ROOT,check=True)
       subprocess.run(['git','commit','--only','-m','Preserve completed race result with immutable admission '+j['name'],'--',*files],cwd=ROOT,check=True)
       subprocess.run(['git','push','origin','main'],cwd=ROOT,check=True)
     except Exception as e:print(json.dumps(dict(run=j['name'],error=repr(e))),flush=True)
  time.sleep(30)
if __name__=='__main__':main()
