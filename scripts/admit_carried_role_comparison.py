"""Admit two native arms after the real paired update gate; no dense control."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];PLAN=ROOT/'experiments/queue/aws_model_improvement_repair_20261005T161000Z'
BASE='aws_g_hier_carried_20261015_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    pre='experiments/results/hierarchy/aws_g_hier_carried_role_gate_20261015_v1.json'
    gate=json.loads((ROOT/pre).read_text());assert gate['status']=='completed'
    old,new=gate['rows']
    for level in map(str,range(1,8)):
        ratio=sum(x[level] for x in new['sensitivity'])/sum(x[level] for x in old['sensitivity'])
        assert ratio>2, (level,ratio)
    assert new['max_rss_kib']*1.5<3000000 and new['paired_update_s']*64*1.5<1800
    declaration=dict(battle='G-HIER',status='validated_changed_composition_and_learning_comparison',
      task_stationarity='Independent stationary synthetic scenes;not a non-stationary online benchmark.',
      strongest_fighter='Acquired full-width relational actor plus persistent version-bound carried interaction,initial gain0.5. Continued acquired relational control isolates benefit beyond further learning.',
      adaptation='64additional paired AdamW1e-3 updates/128presentations per arm;both start from same acquired relational raw weights,wd.01,globalclip1. Whole labels and anonymous TRAIN geometry only;every counterpart exists in TRAIN. Same pair order and common-race coupling. No latent IDs/part labels.',
      state_history='All32raw events,world-pixel state andscene-relative upper memory retained through each scene. Independent scene reset;no truncation. Role views retain latest value/time per source level/quadrant and evolve with native transport.',
      primary='Fixed TRAIN ridge.01 native binary affine head on64TRAIN features,scored on64unseen-combination/nuisance DEV scenes. Joint/scalar heads are diagnostics. Initial/final feature caches retained;TEST not model-scored.',
      retained=['native receipt/value races','separate keys/values','persistent indexed state','local losing-route teacher','native generated views','pure timer interior credit','observed-frame geometry','role-conditioned and cross-child paths'],
      restrictions='Geometric ownership/phase barriers. Role selection is four quadrants per source level,not learned part discovery. Exact queue-order credit absent. Nonzero carrier changes initial predictions. Carrier stays version-bound until native publication and follows native transport. Equal presentations not equal parameters/arithmetic.',
      resources='AWSsix-slot coordinator slots1and6,distinct physical cores,one thread each,RSS3MKiB/VMS12MKiB per arm,8GiB reserve,watchdogs,3600s. Native work,role MAdds/products/transport/logical state and complete optimizer/scoring work recorded separately.',
      decision='Does the measured carried-path sensitivity repair improve held-out composition beyond continued training of the acquired native role actor? Conditional frozen emitted-content audit follows a successful DEV gate;no automatic larger fit.')
    admission=ROOT/f'experiments/hierarchy/{BASE}_admission.json'
    with admission.open('x') as f:json.dump(declaration,f,indent=2);f.write('\n')
    names=list(gate['source_sha256'])+['experiments/hierarchy/carried_role_learning.py','experiments/hierarchy/carried_role_decision.py','scripts/continue_carried_role_comparison.py','experiments/hierarchy/paired_composition_learning.py']+['experiments/theory/ROLE_MEMORY_AND_INTERACTION_DEPTH.md','tests/test_relational_frame_composition.py','tests/test_observed_training_pairs.py','tests/test_paired_composition_recovery.py','scripts/admit_carried_role_comparison.py',str(admission.relative_to(ROOT)),pre]
    bindings={n:sha(ROOT/n) for n in sorted(set(names))};slots={}
    for slot,arm in [('1','continued'),('6','carried')]:
        tag=f'aws_g_hier_carried_{arm}_20261015_v1';q=ROOT/f'experiments/queue/hierarchy/{tag}/{tag}.txt';q.parent.mkdir(parents=True,exist_ok=False)
        q.write_text('# G-HIER: '+declaration['decision']+'\n'+tag+' experiments/hierarchy/carried_role_learning.py --tag '+tag+' --arm '+arm+' --steps 64\n')
        slots[slot]=[dict(name=tag,kind='hierarchy',queue=str(q.relative_to(ROOT)),queue_sha256=sha(q),source_sha256=bindings,result=f'experiments/results/hierarchy/{tag}.json',requires=[dict(result=pre,stage='carried_role_gate')],timeout_s=3600,rss_kb=3000000,vms_kb=12000000)]
    marker=PLAN/'addenda'/('000_'+BASE+'.json')
    with marker.open('x') as f:json.dump(dict(host='ip-172-31-47-132',battle='G-HIER',purpose=declaration['decision'],parent_manifest_sha256=sha(PLAN/'manifest.json'),slots=slots),f,indent=2);f.write('\n')
    print(marker.relative_to(ROOT))
if __name__=='__main__':main()
