"""Admit acquired-model integration only after completed repair contracts."""
import hashlib,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];PLAN=ROOT/'experiments/queue/aws_model_improvement_repair_20261005T161000Z'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert os.uname().nodename=='ip-172-31-47-132'
    gate='experiments/results/diagnostics/aws_r1_review_repairs_20261017_v4.json';d=json.loads((ROOT/gate).read_text());assert d['gate']=='PASS'
    for n,h in d['source_sha256'].items():assert sha(ROOT/n)==h,n
    tag='aws_r1_review_acquired_20261017_v1'
    purpose=dict(battle='R1',stationarity='Chronological acquired-native TRAIN integration gate only; no DEV/TEST fitting or selection.',fighter='Saved best ordinary eight-layer complete-native actor, persistent indexed episodic causes, deferred writer credit, queried and direct producer VJPs, actual nonlinear one-event query-alternative teacher, semantic packed-clock optimizer groups.',online='Score forecast before outcome-driven teaching. Replay alternatives from the same numerical prefix/common hazard clocks, commit only factual state. Four support+four continued TRAIN targets; four discarded exact-recovery targets paid.',state_history='Reuse actor weights, all Adam moments, numeric state and RNGs. Explicitly migrate optimizer grouping without restarting moments. Old publications have no invented direct provenance. Verify legacy native-head law against bound historical source and acquired fixed-weight forecast equality.',retained='Native races/complex temporal memory, indexed state, separate keys/values, persistent versioned queries, dormant concepts, counterfactual routes, producer and episodic writer credit. No dense carrier or recurrent BPTT.',restrictions='One-event nonlinear categorical query credit, not full historical routing/timing or inbox provenance. Existing receiver route teacher remains approximate. Critic policy is not installed/promoted. Eight actual plus four discarded recovery targets; no quality win claimed.',accounting='Existing slot1,one thread,3MKiB RSS/12MKiB VMS,8GiB reserve,7200s timeout; retain peakRSS and replay/snapshot counters. No full arithmetic/energy claim.',decision='Does the combined repair preserve acquired forecasts, parameter objects, Adam moments and recovery while delivering finite direct/query/episodic producer updates in the actual eight-layer model?')
    names=[str(p.relative_to(ROOT)) for p in (ROOT/'sleeping_machines').glob('*.py')]+['experiments/credit/review_acquired_gate.py','experiments/credit/r1_native_reciprocal_future.py','tests/test_deferred_episodic_writer_credit.py','tests/test_native_future_update_credit.py','tests/test_native_query_migration.py','tests/test_native_indexed_cause_head.py','experiments/native_query_migration.py','experiments/r1_composition_credit_pair.py','experiments/r1_local_native_deep.py','experiments/r1_cerl.py']
    q=ROOT/f'experiments/queue/r1/{tag}/{tag}.txt';q.parent.mkdir(parents=True,exist_ok=True);text='# '+json.dumps(purpose)+'\n'+f'{tag} experiments/credit/review_acquired_gate.py --tag {tag}\n'
    if q.exists():assert q.read_text()==text
    else:q.write_text(text)
    job=dict(name=tag,kind='r1',queue=str(q.relative_to(ROOT)),queue_sha256=sha(q),source_sha256={n:sha(ROOT/n) for n in names},result=f'experiments/results/diagnostics/{tag}.json',requires=[dict(result=gate,stage='review repair integrated numerical gate')],timeout_s=7200,rss_kb=3000000,vms_kb=12000000)
    path=PLAN/'addenda/000_aws_r1_review_acquired_20261017_v1.json';packet=dict(host=os.uname().nodename,battle='R1',purpose=purpose,parent_manifest_sha256=sha(PLAN/'manifest.json'),slots={'1':[job]})
    if path.exists():assert json.loads(path.read_text())==packet
    else:
        tmp=path.with_suffix('.writing');tmp.write_text(json.dumps(packet,indent=2)+'\n');tmp.replace(path)
    print(path)
if __name__=='__main__':main()
