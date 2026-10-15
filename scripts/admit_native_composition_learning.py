"""Admit native indexed pooling gate and conditional bounded FIT learning."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sleeping_machines.native_memory_composition import NativeMemoryComposition
PLAN=ROOT/'experiments/queue/aws_model_improvement_repair_20261005T161000Z'
BASE='aws_g_hier_native_composition_20261015_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 declaration=dict(status='validated_for_conditional_pipeline',battle='G-HIER',
  task_stationarity='Annotated independent FIT gestures; first32 raw events/episode. Bounded memorization/feature-use gate, not online or generalization comparison.',
  strongest_fighter='Eight native indexed-temporal-object levels x four stages, followed by root pooling over all active lower learned memories and a native race-generated readout. Reuses protected SplitEventHeads, tensor-time native transport and actual stochastic clocks; no dense/shallow proxy.',
  adaptation='64 batch1 AdamW updates,lr.001,wd.01,one FIT recording only. Exact checkpoints every4updates. No selection/DEV/TEST and no new external reference fit.',
  state_history='32raw events relative to episode boundary,128spatial resolution retained. Persistent addressed state through input events, multi-memory queries and generated readout. Independent episode reset only; no within-episode memory truncation.',
  retained=['native delays and races','protected+decaying/rotating memory','parallel heads','separate keys/values','selected addressed writes','local counterfactual route teacher','tensor input/clock derivatives','indexed all-lower-learned-memory reads','typed query responses','race-generated readout','causal publication/versioning','shared rule capacity across spatial objects'],
  missing=['exact future queue-order alternative credit','learned sparse query/arming/coalescing policy','learned parent ownership','content-indexed discovery','full FLOP/traffic/optimizer ledger','held-out semantic transfer'],
  restrictions='32 raw events and one FIT recording are explicit learnability limits, not family benchmark evidence. All active lower learned memories in the requested root are queried once; raw-input shortcut excluded. Geometric ownership, chronological CPU heap and virtual clock units remain development choices.',
  resources='Existing coordinator slot6,one core/thread,RSS3000000KiB,VMS12000000KiB,8GiB reserve,watchdog. Preflight measures intended batch1 and requires1.25RSSmargin and64update1.5time margin under1200s. Learning timeout1800s; no unguarded launch.',
  decision='Can native32-stage multi-memory composition learn input-dependent FIT label association under faithful stochastic execution? Separate class-frequency fitting using entropy/majority and association diagnostics before any scale.',
  protocol='Two fixed evaluation noise tapes shared across all episodes; no per-item random identity. Train/eval bitwise parity and input collision checks precede fit. Model inputs exclude labels/absolute recording time.')
 admission=ROOT/f'experiments/hierarchy/{BASE}_admission.json'
 with admission.open('x') as f:json.dump(declaration,f,indent=2);f.write('\n')
 names={str(Path(m.__file__).resolve().relative_to(ROOT)) for name,m in tuple(sys.modules.items()) if name.startswith('sleeping_machines') and getattr(m,'__file__',None) and str(Path(m.__file__).resolve()).startswith(str(ROOT))}
 names.update(['experiments/hierarchy/native_composition_data.py','experiments/hierarchy/native_composition_learning.py','tests/test_native_memory_composition.py','experiments/theory/NATIVE_MULTI_MEMORY_COMPOSITION.md','scripts/admit_native_composition_learning.py',str(admission.relative_to(ROOT))])
 bindings={n:sha(ROOT/n) for n in sorted(names)};jobs=[]
 dataresult=f'experiments/results/hierarchy/{BASE}_data.json'
 for stage in ['data','learn']:
  tag=BASE+'_'+stage;queue=ROOT/f'experiments/queue/hierarchy/{tag}/{tag}.txt';queue.parent.mkdir(parents=True,exist_ok=False)
  command=f'experiments/hierarchy/native_composition_data.py --tag {tag}' if stage=='data' else f'experiments/hierarchy/native_composition_learning.py --tag {tag} --preflight {dataresult} --steps 64'
  with queue.open('x') as f:f.write('# G-HIER: '+declaration['decision']+'\n'+tag+' '+command+'\n')
  requires=[dict(result='experiments/results/diagnostics/native_multi_memory_contracts_20261015_v1.json',stage='development_contracts')] if stage=='data' else [dict(result=dataresult,stage='native_composition_data',rss_margin=1.25)]
  jobs.append(dict(name=tag,kind='hierarchy',queue=str(queue.relative_to(ROOT)),queue_sha256=sha(queue),source_sha256=bindings,result=f'experiments/results/hierarchy/{tag}.json',requires=requires,timeout_s=1800,rss_kb=3000000,vms_kb=12000000))
 marker=PLAN/'addenda'/('000_'+BASE+'.json')
 with marker.open('x') as f:json.dump(dict(host='ip-172-31-47-132',battle='G-HIER',purpose=declaration['decision'],parent_manifest_sha256=sha(PLAN/'manifest.json'),slots={'6':jobs}),f,indent=2);f.write('\n')
 print(json.dumps(dict(jobs=[j['name'] for j in jobs],slot=6)))
if __name__=='__main__':main()
