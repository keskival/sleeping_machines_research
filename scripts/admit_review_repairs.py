"""Admit a uniquely source-bound R1 review repair gate in the existing gym."""
import argparse,hashlib,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];PLAN=ROOT/'experiments/queue/aws_model_improvement_repair_20261005T161000Z'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--version',required=True);p.add_argument('--slot',choices=list('123456'),default='1');p.add_argument('--tests',nargs='+',default=['tests/test_review_repairs.py']);a=p.parse_args();assert a.version.startswith('v') and a.version[1:].isdigit();assert os.uname().nodename=='ip-172-31-47-132'
    tag=f'aws_r1_review_repairs_20261017_{a.version}'
    purpose=dict(battle='R1',stationarity='Numerical repair and integrated learner contracts, no benchmark fitting or DEV/TEST use.',fighter='Native sampled producer credit now includes direct version-bound input edges; stable feature-epoch critic; live first-order writer-input VJP; nonlinear task-loss route teaching where covered by named tests.',online='Tests reveal outcomes only after their factual forecast; local optimizer steps and serialized recovery checked. No competitive online win claim.',state_history='Factual numerical predictions and RNGs retained; exact producer-version edges retained by GC. No active trainer/checkpoint/source changed.',retained='Native temporal races, persistent receivers, queries, separate keys/values and counterfactual credit. Critic feature parameters fixed within a calibration epoch; numerical state continues to evolve.',restrictions='Tiny finite DAG/real learner contracts. Missing temporal/inbox provenance and historical timing/route derivatives remain explicit; no whole-history exactness.',accounting='Existing idle slot1,one thread,3MKiB RSS/12MKiB VMS,8GiB reserve,1800s timeout. Native replay and gradient/storage costs are not hardware gains.',decision='Do the repairs close the identified conditional credit/coordinate defects, preserve factual computation and serialize exactly before any new fitting admission?')
    names=[str(p.relative_to(ROOT)) for p in (ROOT/'sleeping_machines').glob('*.py')]+a.tests+['experiments/credit/review_repair_gate.py','tests/rmr_writer_reference.py','experiments/r1_rmr.py','experiments/r1_cerl.py','experiments/native_slot_memory.py','experiments/r1_deep_keyed_lm.py','experiments/r1_local_native_deep.py']
    q=ROOT/f'experiments/queue/r1/{tag}/{tag}.txt';q.parent.mkdir(parents=True,exist_ok=True);text='# '+json.dumps(purpose)+'\n'+f'{tag} experiments/credit/review_repair_gate.py --tag {tag} --tests '+' '.join(a.tests)+'\n'
    if q.exists():assert q.read_text()==text
    else:q.write_text(text)
    job=dict(name=tag,kind='r1',queue=str(q.relative_to(ROOT)),queue_sha256=sha(q),source_sha256={n:sha(ROOT/n) for n in names},result=f'experiments/results/diagnostics/{tag}.json',requires=[],timeout_s=1800,rss_kb=3000000,vms_kb=12000000)
    path=PLAN/f'addenda/000_aws_r1_review_repairs_20261017_{a.version}.json';packet=dict(host=os.uname().nodename,battle='R1',purpose=purpose,parent_manifest_sha256=sha(PLAN/'manifest.json'),slots={a.slot:[job]})
    if path.exists():assert json.loads(path.read_text())==packet
    else:
        tmp=path.with_suffix('.writing');tmp.write_text(json.dumps(packet,indent=2)+'\n');tmp.replace(path)
    print(path)
if __name__=='__main__':main()
