"""Admit one bounded whole-label composition development run after measured gate."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/'experiments/queue/aws_model_improvement_repair_20261005T161000Z'
TAG='aws_g_hier_coalesced_readout_20261015_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    pre='experiments/results/hierarchy/aws_g_hier_coalesced_preflight_20261015_v2.json'
    gate=json.loads((ROOT/pre).read_text());assert gate['status']=='completed'
    point=gate['measurements'][1]
    assert point['model']=='CoalescedNativeComposition' and point['max_rss_kib']*1.5<3000000
    assert all(n>0 for n in point['content_gradient_norms'])
    declaration=dict(battle='G-HIER',status='validated_frozen_readout_audit',
        stationarity='Stationary synthetic composition scenes, not an online-stream comparison.',
        strongest_fighter='Completed initial/final32-stage coalesced native cores; only native external affine head fitted. Persistent memory,races,keys/values,relative pose,timed emissions retained.',
        adaptation='No native backbone update.64TRAIN scenes fit fixed ridge.01 logistic readout; TRAIN normalization frozen for64DEV scenes. No DEV hyperparameter selection or TEST scoring.',
        state_history='All32 raw events and scene-local numerical memory retained through composition; independent scene reset, one common scoring tape.',
        restrictions='Explicit level barriers/geometric ownership/coalesced emissions; no learned query/emission policy or order-boundary teacher. No auxiliary part identities.',
        resources='Existing AWSslot6 one pinned core/thread,RSS3MKiB,VMS12MKiB,8GiBreserve,watchdogs,1200s. Native features cached; no unchanged refit.',
        decision='Does trained native representation transfer under a well-fitted frozen head, or is failure in representation/generalization? Compare acquired initial core; emitted-content intervention only when DEV beats prior.')
    path=ROOT/f'experiments/hierarchy/{TAG}_admission.json'
    with path.open('x') as f:json.dump(declaration,f,indent=2);f.write('\n')
    names=list(gate['source_sha256'])
    names+=['experiments/hierarchy/coalesced_frozen_readout.py','experiments/hierarchy/coalesced_learning.py','sleeping_machines/convex_native_readout.py','tests/test_convex_native_readout.py','experiments/theory/CONVEX_COMPOSITION_READOUT_AUDIT.md','sleeping_machines/prior_centered_readout.py','experiments/theory/COMPOSITION_MESSAGE_SUFFICIENCY.md','experiments/theory/COALESCED_NATIVE_COMPOSITION.md','scripts/admit_coalesced_readout_audit.py',str(path.relative_to(ROOT)),pre]
    q=ROOT/f'experiments/queue/hierarchy/{TAG}/{TAG}.txt';q.parent.mkdir(parents=True,exist_ok=False)
    q.write_text('# G-HIER: '+declaration['decision']+'\n'+TAG+' experiments/hierarchy/coalesced_frozen_readout.py --tag '+TAG+'\n')
    job=dict(name=TAG,kind='hierarchy',queue=str(q.relative_to(ROOT)),queue_sha256=sha(q),source_sha256={n:sha(ROOT/n) for n in sorted(set(names))},result=f'experiments/results/hierarchy/{TAG}.json',requires=[dict(result='experiments/results/hierarchy/aws_g_hier_coalesced_learning_20261015_v1.json',stage='coalesced_composition_learning')],timeout_s=1200,rss_kb=3000000,vms_kb=12000000)
    marker=PLAN/'addenda'/('000_'+TAG+'.json')
    with marker.open('x') as f:json.dump(dict(host='ip-172-31-47-132',battle='G-HIER',purpose=declaration['decision'],parent_manifest_sha256=sha(PLAN/'manifest.json'),slots={'6':[job]}),f,indent=2);f.write('\n')
    print(marker.relative_to(ROOT))
if __name__=='__main__':main()
