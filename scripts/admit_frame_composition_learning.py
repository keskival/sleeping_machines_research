"""Admit one bounded whole-label composition development run after measured gate."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/'experiments/queue/aws_model_improvement_repair_20261005T161000Z'
TAG='aws_g_hier_frame_learning_20261015_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    pre='experiments/results/hierarchy/aws_g_hier_frame_preflight_20261015_v1.json'
    gate=json.loads((ROOT/pre).read_text());assert gate['status']=='completed'
    point=gate
    assert point['stage']=='frame_preflight' and point['exact_translation_feature_equality'] and point['max_rss_kib']*1.5<3000000
    assert all(n>0 for n in point['content_gradient_norms'])
    declaration=dict(battle='G-HIER',status='validated_bounded_learning',
        stationarity='Stationary independent synthetic scenes. Not a non-stationary/online benchmark.',
        strongest_fighter='32-native-stage/eight-level world-pixel native memory and observed-frame parent composition. Same acquired core; measured translation failure removed exactly, combined nuisance displacement .416184→.003158, all-level gradients. Cost4800candidates/150deliveries, no quality inferred.',
        adaptation='128 AdamW1e-3 whole-scene TRAIN-only updates, batch1, seed211; acquired gesture core reused, incompatible external 11-class head replaced with freshly initialized 2-class head. Globalclip1 and wd.01 disclosed.',
        state_history='All32 events per scene, native numerical state throughout raw integration and composition; resets only between independent scenes. No gradient truncation inside scene.',
        data='128 TRAIN scenes/32 pairs;64 calibration TRAIN scenes,64 unseen-combination DEV scenes. Every child code supported; nuisance tapes disjoint. No hidden part IDs/supervision enter model. TEST excluded from model scoring.',
        selection='No model selection inside fit. Initial/final diagnostics and cached features; fixed TRAIN ridge.01 full-head diagnostic on both cores; centre/prior/nonnegative gain fitted on TRAIN only. One common scoring tape113 across inputs. If DEV beats prior, zero emitted-content intervention with same frozen calibration; combination-cluster paired bootstrap.',
        retained=['native temporal races','persistent addressed state','separate keys/values','local losing-route credit','all eight levels/four stages','relative pose','race-generated child views','pure timer interior credit'],
        restrictions='Explicit scene completion and level drain barriers; no learned emission/query/ownership policy, no event-order boundary credit. World-pixel memories persist; frame committed from revealed scene bounds, root support128pixels. Intermediate upward views are coalesced; this can lose fine change signals. No external control trained.',
        resources='AWScoordinator slot6 one physicalcore/thread,RSS3000000KiB,VMS12000000KiB,8GiB reserve,watchdogs,3600s. Exact model/Adam/RNG/order checkpoints every8updates. Native work counters by phase; no total-FLOP/hardware claim.',
        decision='Does learned composition transfer to unfamiliar combinations and depend on emitted content? If not, diagnose binding/credit; short fit is development, not family loss. No automatic grid or TEST scoring.')
    path=ROOT/f'experiments/hierarchy/{TAG}_admission.json'
    with path.open('x') as f:json.dump(declaration,f,indent=2);f.write('\n')
    names=list(gate['source_sha256'])
    names+=['experiments/hierarchy/frame_learning.py','sleeping_machines/prior_centered_readout.py','sleeping_machines/convex_native_readout.py','tests/test_frame_native_composition.py','experiments/theory/COMPOSITION_MESSAGE_SUFFICIENCY.md','experiments/theory/COALESCED_NATIVE_COMPOSITION.md','scripts/admit_frame_composition_learning.py',str(path.relative_to(ROOT)),pre]
    q=ROOT/f'experiments/queue/hierarchy/{TAG}/{TAG}.txt';q.parent.mkdir(parents=True,exist_ok=False)
    q.write_text('# G-HIER: '+declaration['decision']+'\n'+TAG+' experiments/hierarchy/frame_learning.py --tag '+TAG+' --steps 128\n')
    job=dict(name=TAG,kind='hierarchy',queue=str(q.relative_to(ROOT)),queue_sha256=sha(q),source_sha256={n:sha(ROOT/n) for n in sorted(set(names))},result=f'experiments/results/hierarchy/{TAG}.json',requires=[dict(result=pre,stage='frame_preflight')],timeout_s=3600,rss_kb=3000000,vms_kb=12000000)
    marker=PLAN/'addenda'/('000_'+TAG+'.json')
    with marker.open('x') as f:json.dump(dict(host='ip-172-31-47-132',battle='G-HIER',purpose=declaration['decision'],parent_manifest_sha256=sha(PLAN/'manifest.json'),slots={'6':[job]}),f,indent=2);f.write('\n')
    print(marker.relative_to(ROOT))
if __name__=='__main__':main()
