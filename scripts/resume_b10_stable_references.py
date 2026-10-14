"""Bind preserved fits to stable common head and charge interruption/replay work."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments/market'))
from b10_matched_pipeline import sha

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--gate',required=True);a=ap.parse_args()
 receipt=json.loads((ROOT/a.gate).read_text());assert receipt['training_admitted'] and receipt['status']=='completed'
 assert all(sha(ROOT/name)==digest for name,digest in receipt['source_sha256'].items())
 original_path=ROOT/'experiments/results/market/aws_b10_matched_admission_20261014_v5.json';original=json.loads(original_path.read_text());receipt['resume_bindings']={};receipt['fresh_interruption_charge']={}
 # New admission smokes and numerical-failure replay are additional work. Use
 # whole-update upper bounds when failing partial backwards lack a full ledger.
 diagnostics={'native':0,'lstm':14,'transformer':0}
 trace=ROOT/'experiments/results/market/aws_b10_transformer_failure_audit_20261014_v1.json'
 if trace.exists():diagnostics['transformer']=len(json.loads(trace.read_text())['updates'])+1
 for seed in [0,1,2]:
  for family in ['native','lstm','transformer']:
   old_prefix=f'aws_b10_matched_{family}_s{seed}_20261014_v2';checkpoint=f'experiments/results/market/{old_prefix}.progress.pt'
   new_cost=receipt['update_arithmetic'][family];old_cost=original['update_arithmetic'][family]
   # At most one logical epoch can lie beyond the last committed checkpoint.
   # This conservative ceiling accounts for interrupted/discarded fitting.
   epoch=50*max(new_cost['16'],old_cost['16'])+max(new_cost['1'],old_cost['1'])
   charge=receipt['overhead_arithmetic'][family]+epoch+diagnostics[family]*max(new_cost['16'],old_cost['16'])
   if (ROOT/checkpoint).exists():
    receipt['resume_bindings'][family+':'+str(seed)]=dict(checkpoint=checkpoint,checkpoint_sha256=sha(ROOT/checkpoint),old_admission_sha256=sha(original_path),additional_charged_arithmetic=charge,charge_convention='conservative one-epoch discarded-update ceiling plus recost-gates and familydiagnostic update ceilings; repeated engineering charge per seed',old_program_source_sha256=original['source_sha256'])
   else:receipt['fresh_interruption_charge'][family+':'+str(seed)]=epoch+diagnostics[family]*max(new_cost['16'],old_cost['16'])
 receipt['head_upgrade']='Same normalized recording-cell law evaluated through log_sigma; finite-range parity and failing-data replay validated'
 receipt['input_sha256']={a.gate:sha(ROOT/a.gate),str(original_path.relative_to(ROOT)):sha(original_path),'experiments/results/market/aws_b10_shared_numerical_pause_20261014_v1.json':sha(ROOT/'experiments/results/market/aws_b10_shared_numerical_pause_20261014_v1.json')}
 admission='experiments/results/market/aws_b10_stable_resume_admission_20261014_v1.json';out=ROOT/admission;assert not out.exists();out.write_text(json.dumps(receipt,indent=2)+'\n')
 plan=ROOT/'experiments/queue/aws_model_improvement_repair_20261005T161000Z';slots={};assignment=[4,3,6,1,3,4,1,6,4];index=0
 for seed in [0,1,2]:
  for family in ['native','lstm','transformer']:
   tag=f'aws_b10_stable_{family}_s{seed}_20261014_v1';q=ROOT/f'experiments/queue/aws_b10_matched_controls_20261014_v1/{tag}/{tag}.txt';q.parent.mkdir(exist_ok=False)
   args=f'--tag {tag} --family {family} --seed {seed} --admission {admission}'
   if family+':'+str(seed) in receipt['resume_bindings']:args+=' --resume-from '+receipt['resume_bindings'][family+':'+str(seed)]['checkpoint']
   q.write_text('# B10: samehead logscale numerical repair, savedactor/Adam/RNG/data recovery; decision full DEV matchedchargedarithmetic\n'+f'{tag} experiments/market/b10_matched_fit.py {args}\n')
   job=dict(name=tag,kind='b10_job',queue=str(q.relative_to(ROOT)),queue_sha256=sha(q),source_sha256=receipt['source_sha256'],result=f'experiments/results/market/{tag}.json',requires=[dict(result=admission)],timeout_s=receipt['timeouts_s'][family],rss_kb=3000000,vms_kb=12000000)
   slots.setdefault(str(assignment[index]),[]).append(job);index+=1
 p=plan/'addenda'/f'zzzzzzzzzzzzzzzzzzzzzzzzzzzzb10_{time.time_ns()}_stable_full.json';p.write_text(json.dumps(dict(host='ip-172-31-47-132',battle='B10',purpose='Repairedcommonhead reference resume; preserve acquiredstate and conservativelycharge wasted/replaywork',parent_manifest_sha256=sha(plan/'manifest.json'),slots=slots),indent=2)+'\n')
 order=ROOT/'experiments/queue/aws_b10_matched_controls_20261014_v1/work_order.json';w=json.loads(order.read_text());w.update(status='stable_common_head_full_fits_runtime_admitted',stable_admission=admission,stable_addendum=str(p.relative_to(ROOT)));order.write_text(json.dumps(w,indent=2)+'\n');print(p)
if __name__=='__main__':main()
