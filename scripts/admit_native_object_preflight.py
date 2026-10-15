"""Admit a bounded raw-event native-memory preflight; no optimizer/fit."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sleeping_machines.native_object_hierarchy import NativeObjectHierarchy
TAG='aws_g_hier_native_objects_preflight_20261015_v1'
PLAN=ROOT/'experiments/queue/aws_model_improvement_repair_20261005T161000Z'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 declaration=dict(status='validated_for_preflight_only',battle='G-HIER',
  task_stationarity='Episode classification development;32 raw events from first official FIT gesture; diagnostic only',
  strongest_fighter='Eight levels of indexed temporal objects, four native SplitEventHeads stages/level, protected+dynamic memory, parallel races and tensor-time transport. Reuses native event learner rather than adding a race to synchronous pooling.',
  adaptation='No optimizer or fitting; one real-input loss/backward checks native signal paths. No generalization or online verdict.',
  state_history='Persistent addressed state and separately published views; raw timestamps retained, no time bins or per-layer resets; indexed reads and query/timer outputs preserve input records.',
  retained=['native computational delays','temporal state decay/rotation','protected persistent modes','parallel native races','separate keys/values','selected addressed writes','local native counterfactual teacher','tensor arrival derivatives across levels','indexed lower-memory query','race-generated emission','versioned causal publication','bounded per-stream retry state'],
  missing=['exact queue-reordering counterfactual credit','learned query/arming/coalescing policy','learned object ownership','content-indexed discovery','complete work/traffic ledger','trained semantic transfer'],
  restrictions='Known geometric parent addressing, one-source common parent receiver with typed source/pose content, chronological CPU heap reference.32 raw events is a bounded resource diagnostic, never a quality verdict. Native local credit is not exact full-event-graph credit.',
  resources='Existing coordinator slot6, one core/thread, RSS3000000KiB,VMS12000000KiB,at least8GiB MemAvailable, watchdog,1800s. Virtual model time is not hardware latency.',
  decision='Does the native memory+generator hierarchy handle raw event updates, indexed queries and timer outputs with finite gradients and bounded measured resources? Select native learning engineering next; no automatic long fit.')
 admission=ROOT/f'experiments/hierarchy/{TAG}_admission.json'
 with admission.open('x') as f:json.dump(declaration,f,indent=2);f.write('\n')
 names={str(Path(m.__file__).resolve().relative_to(ROOT)) for name,m in tuple(sys.modules.items()) if name.startswith('sleeping_machines') and getattr(m,'__file__',None) and str(Path(m.__file__).resolve()).startswith(str(ROOT))}
 names.update(['experiments/hierarchy/native_object_preflight.py','scripts/admit_native_object_preflight.py','tests/test_indexed_temporal_object.py','tests/test_native_object_hierarchy.py','experiments/theory/INDEXED_TEMPORAL_OBJECTS.md',str(admission.relative_to(ROOT))])
 bindings={n:sha(ROOT/n) for n in sorted(names)}
 queue=ROOT/f'experiments/queue/hierarchy/{TAG}/{TAG}.txt';queue.parent.mkdir(parents=True,exist_ok=False)
 with queue.open('x') as f:f.write('# G-HIER: '+declaration['decision']+'\n'+TAG+' experiments/hierarchy/native_object_preflight.py --tag '+TAG+'\n')
 job=dict(name=TAG,kind='hierarchy',queue=str(queue.relative_to(ROOT)),queue_sha256=sha(queue),source_sha256=bindings,result=f'experiments/results/hierarchy/{TAG}.json',
  requires=[dict(result='experiments/results/diagnostics/native_temporal_object_contracts_20261015_v1.json',stage='development_contracts')],timeout_s=1800,rss_kb=3000000,vms_kb=12000000)
 marker=PLAN/'addenda'/('000_aws_g_hier_aaa_native_objects_preflight_20261015_v1.json')
 with marker.open('x') as f:json.dump(dict(host='ip-172-31-47-132',battle='G-HIER',purpose=declaration['decision'],parent_manifest_sha256=sha(PLAN/'manifest.json'),slots={'6':[job]}),f,indent=2);f.write('\n')
 print(json.dumps(dict(job=TAG,slot=6,source_bindings=len(bindings))))
if __name__=='__main__':main()
