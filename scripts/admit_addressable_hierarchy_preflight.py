"""Admit one no-fit real-TRAIN G-HIER preflight, preserving active jobs."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
TAG='aws_g_hier_addressable_preflight_20261015_v1'
PLAN=ROOT/'experiments/queue/aws_model_improvement_repair_20261005T161000Z'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    declaration=dict(battle='G-HIER',status='validated_for_preflight_only',
        task_stationarity='Subject-disjoint gesture recognition development; preflight uses first FIT recording only',
        strongest_fighter='32-cell persistent role-binding parent with all-lower-level addressed memories, relation composition and full-continuation categorical credit; repairs immediate-only read restriction',
        adaptation='No optimizer or fit; factual backward and route audit validate learning paths only',
        state_history='Fixed first1s/8bins/128x128 event evidence; all-level state and input retained without hidden detach; refinement can reread evidence without new input',
        retained=['hard exponential races','separate keys/values','parallel child roles','relation composition','temporal input surfaces','persistent all-level memory','delayed upper-state query conditioning','selected-content transport','full-continuation categorical credit','cell-balanced audit selection'],
        missing=['raw asynchronous event dispatch','content-indexed discovery','dynamic parent ownership','explicit counterfactual future optimizer credit','zero losing-value execution','full FLOP ledger','trained compositional generalization'],
        restrictions='Dense tensor/gather backend, fixed stencils9/source and binned event adapter; no online-stream, latency or energy verdict',
        resources='CPU-only, existing six-slot coordinator slot6, one core/thread, RSS3000000KiB, VMS12000000KiB, at least8GiB MemAvailable, watchdog, timeout1800s',
        decision='Does128-resolution strongest parent construction fit the reserved resource envelope and expose finite real-input learning paths? No automatic fit admission')
    admission=ROOT/f'experiments/hierarchy/{TAG}_admission.json'
    with admission.open('x') as f:json.dump(declaration,f,indent=2);f.write('\n')
    names=['sleeping_machines/addressable_parent_hierarchy.py','sleeping_machines/parent_race_hierarchy.py',
        'sleeping_machines/spatial_race_hierarchy.py','experiments/hierarchy/visual_representation.py',
        'experiments/hierarchy/addressable_preflight.py','tests/test_addressable_parent_hierarchy.py',
        'experiments/theory/ADDRESSABLE_COMPOSITION_MEMORY.md','scripts/admit_addressable_hierarchy_preflight.py',
        str(admission.relative_to(ROOT))]
    bindings={n:sha(ROOT/n) for n in names}
    queue=ROOT/f'experiments/queue/hierarchy/{TAG}/{TAG}.txt';queue.parent.mkdir(parents=True,exist_ok=False)
    with queue.open('x') as f:f.write('# G-HIER: '+declaration['decision']+'\n'+TAG+' experiments/hierarchy/addressable_preflight.py --tag '+TAG+'\n')
    job=dict(name=TAG,kind='hierarchy',queue=str(queue.relative_to(ROOT)),queue_sha256=sha(queue),
       source_sha256=bindings,result=f'experiments/results/hierarchy/{TAG}.json',requires=[],timeout_s=1800,rss_kb=3000000,vms_kb=12000000)
    packet=dict(host='ip-172-31-47-132',battle='G-HIER',purpose=declaration['decision'],
       parent_manifest_sha256=sha(PLAN/'manifest.json'),slots={'6':[job]})
    marker=PLAN/'addenda'/('000_'+TAG+'.json')
    with marker.open('x') as f:json.dump(packet,f,indent=2);f.write('\n')
    print(json.dumps(dict(queue=str(queue.relative_to(ROOT)),slot=6,admission=str(admission.relative_to(ROOT)))))
if __name__=='__main__':main()
