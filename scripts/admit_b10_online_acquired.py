"""Read-only readiness watcher that admits already-authorized online comparisons.
Never trains; all numerical work remains with the bounded slot scheduler.
"""
import argparse,hashlib,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');a=p.parse_args();plan=ROOT/'experiments/queue/aws_model_improvement_repair_20261005T161000Z';done=set()
 while True:
  admission=ROOT/'experiments/market/B10_ONLINE_CAPABILITY_ADMISSION.json';r=json.loads(admission.read_text());gate=ROOT/r['gate_result']
  if gate.exists() and json.loads(gate.read_text()).get('status')=='completed':
   for family,slot in [('native',4),('lstm',3)]:
    tag=f'aws_b10_online_{family}_s0_20261014_v1';receipt=ROOT/f'experiments/results/market/aws_b10_stable_{family}_s0_20261014_v1.json';marker=plan/f'addenda/zzzzzzzzzzzzzzzzzzzzzzzzzzzzb10_online_{family}_20261014_v1.json'
    if marker.exists():done.add(family);continue
    if not receipt.exists() or json.loads(receipt.read_text()).get('status')!='completed':continue
    for n,d in r['source_sha256'].items():assert sha(ROOT/n)==d,n
    q=ROOT/f'experiments/queue/b10_online/{tag}/{tag}.txt';q.parent.mkdir(parents=True,exist_ok=True);q.write_text(f'# B10: native strengths and online reference adaptation on the same chronological stream; complete work charged\n{tag} experiments/market/b10_online_comparison.py --tag {tag} --mode online --admission {admission.relative_to(ROOT)} --family {family}\n');sources={**r['source_sha256'],str(admission.relative_to(ROOT)):sha(admission)};job=dict(name=tag,kind='b10_job',queue=str(q.relative_to(ROOT)),queue_sha256=sha(q),source_sha256=sources,result=f'experiments/results/market/{tag}.json',requires=[{'result':r['gate_result'],'rss_margin':1.25},{'result':str(receipt.relative_to(ROOT))}],timeout_s=10800,rss_kb=3000000,vms_kb=12000000);packet=dict(host='ip-172-31-47-132',battle='B10',purpose='Online comparison reuses completed acquired actor; no duplicate offline fitting',parent_manifest_sha256=sha(plan/'manifest.json'),slots={str(slot):[job]});tmp=marker.with_suffix('.tmp');tmp.write_text(json.dumps(packet,indent=2)+'\n');tmp.replace(marker);done.add(family);print('Admitted '+tag,flush=True)
  if len(done)==2 or not a.watch:return
  time.sleep(30)
if __name__=='__main__':main()
