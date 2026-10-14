"""Admit nine full fits from a completed source-bound B10 budget receipt."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments/market'))
from b10_matched_pipeline import sha

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--admission',required=True);a=ap.parse_args()
 receipt=json.loads((ROOT/a.admission).read_text());assert receipt['status']=='completed' and receipt['training_admitted']
 assert all(sha(ROOT/name)==digest for name,digest in receipt['source_sha256'].items())
 assert receipt['max_rss_kb']<3000000
 plan=ROOT/'experiments/queue/aws_model_improvement_repair_20261005T161000Z';order=ROOT/'experiments/queue/aws_b10_matched_controls_20261014_v1/work_order.json';slots={}
 # Preserve the long acquisition on slot2; use currently free4/5 and soon-free1/3/6.
 assignment=[4,5,6,1,3,4,5,6,1];index=0
 for seed in [0,1,2]:
  for family in ['native','lstm','transformer']:
   tag=f'aws_b10_matched_{family}_s{seed}_20261014_v2';q=order.parent/tag/(tag+'.txt');q.parent.mkdir(exist_ok=False)
   q.write_text('# B10: full native/LSTM/Transformer matched arithmetic; decision joint DEV NLL and whole-fit resources at frozen common B\n'+f'{tag} experiments/market/b10_matched_fit.py --tag {tag} --family {family} --seed {seed} --admission {a.admission}\n')
   job=dict(name=tag,kind='b10_job',queue=str(q.relative_to(ROOT)),queue_sha256=sha(q),source_sha256=receipt['source_sha256'],result=f'experiments/results/market/{tag}.json',requires=[dict(result=a.admission)],timeout_s=receipt['timeouts_s'][family],rss_kb=3000000,vms_kb=12000000)
   slots.setdefault(str(assignment[index]),[]).append(job);index+=1
 packet=dict(host='ip-172-31-47-132',battle='B10',purpose='Founder-authorized full native/LSTM/Transformer three-seed matched-arithmetic references; September sealed',parent_manifest_sha256=sha(plan/'manifest.json'),slots=slots)
 p=plan/'addenda'/f'zzzzzzzzzzzzzzzzzzzzzzzzzzzzb10_{time.time_ns()}_full.json';p.write_text(json.dumps(packet,indent=2)+'\n')
 w=json.loads(order.read_text());w.update(status='nine_full_fits_runtime_admitted',training_admitted=True,admission=a.admission,budget_arithmetic=receipt['budget_arithmetic'],full_fits_addendum=str(p.relative_to(ROOT)));order.write_text(json.dumps(w,indent=2)+'\n');print(p)
if __name__=='__main__':main()
