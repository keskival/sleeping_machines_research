"""R1 acquired native composition-credit gate; no new long acquisition."""
import hashlib,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/'experiments/queue/aws_model_improvement_repair_20261005T161000Z'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 assert os.uname().nodename=='ip-172-31-47-132'
 assert int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))>8*1024*1024
 tag='aws_r1_compositional_producer_credit_gate_20261016_v1'
 sources=[str(p.relative_to(ROOT)) for p in (ROOT/'sleeping_machines').glob('*.py')]
 sources+=['experiments/hierarchy/compositional_producer_credit_gate.py','experiments/hierarchy/native_composition_data.py','tests/test_compositional_producer_credit.py','tests/test_coupled_native_query_credit.py','tests/test_native_query_migration.py','experiments/native_query_migration.py','experiments/r1_local_native_deep.py','experiments/r1_cerl.py','experiments/theory/COMPOSITION_LEARNING_CONTRACT.md']
 purpose=dict(battle='R1',stationarity='Stationary FineWeb numerical TRAIN integration only; not a benchmark or online win.',fighter='Private copy of acquired eight-layer fully native learner, indexed bank, numerical eligibility, legal joint queries, local producer credit plus bounded actual-child content VJP. No SSM/attention/MLP carrier.',adaptation='Two8target chronological AdamW updates and one discarded8target exact-resume audit; weights/Adam/numerical state/RNGs restored from acquired checkpoint. No corpus refit, DEV/TEST or automatic continuation.',state_history='All receiver/index/inbox/binding/producer state persists. New lineage only for new publications; old memories remain accessible without invented provenance. Root16/layer; transitive lineage charged.',retained='Temporal races, native complex transport, separate keys/values, sparse selected writes, legal joint alternatives, dormant concepts, identity/role composition, own-write eligibility and delayed producer credit.',restrictions='At most2child edges/path and6child edges/root. Historical choice/timing/base-input/inbox provenance fixed; changing-parameter replay approximate. All stages and declared keys scored; synchronous optimizer/clipping. Conditional finite-DAG content credit, not exact full-history gradient.',accounting='Slot3 through existing six-slot coordinator after active work; one thread,3MKiB RSS/12MKiB VMS,8GiB reserve,1800s timeout. Charge16TRAINtargets and8discarded audit, tests, checkpoint load, child reads, transport, credit, lineage, optimizer and RSS. No energy/latency or complete FLOP claim.',decision='Does delayed upper loss reach actual lower composers with independent finite-DAG VJP agreement, identical factual/RNG semantics, bounded work and exact recovery? Numerical gate only; scaling orders stay primary.')
 q=ROOT/f'experiments/queue/r1/{tag}/{tag}.txt';q.parent.mkdir(parents=True,exist_ok=True)
 text='# '+json.dumps(purpose)+'\n'+tag+' experiments/hierarchy/compositional_producer_credit_gate.py --tag '+tag+'\n'
 if q.exists():assert q.read_text()==text
 else:q.write_text(text)
 job=dict(name=tag,kind='r1',queue=str(q.relative_to(ROOT)),queue_sha256=sha(q),source_sha256={s:sha(ROOT/s) for s in sorted(set(sources))},result=f'experiments/results/diagnostics/{tag}.json',requires=[],timeout_s=1800,rss_kb=3000000,vms_kb=12000000)
 out=PLAN/'addenda/000_aws_r1_000_composition_learning_contract_20261016_v1.json'
 packet=dict(host=os.uname().nodename,battle='R1',purpose=purpose,parent_manifest_sha256=sha(PLAN/'manifest.json'),slots={'3':[job]})
 if out.exists():assert json.loads(out.read_text())==packet
 else:
  tmp=out.with_suffix('.writing');tmp.write_text(json.dumps(packet,indent=2)+'\n');tmp.replace(out)
 print('Admitted numerical acquired-native composition gate; no new long fit.')
if __name__=='__main__':main()
