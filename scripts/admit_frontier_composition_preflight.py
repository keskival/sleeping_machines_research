"""One no-fit G-HIER resource gate; no conditional fit is admitted."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/'experiments/queue/aws_model_improvement_repair_20261005T161000Z'
TAG='aws_g_hier_frontier_preflight_20261015_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    declaration=dict(battle='G-HIER',status='validated_no_fit_gate',
        task='Stationary controlled independent scenes; three nested spatial arrangements, unseen combinations and split-disjoint nuisance tapes. No online-stream verdict.',
        strongest_fighter='Acquired 32-native-stage/eight-level memory compositor; compare existing all-lower read with explicit adjacent stabilized frontiers, preserving identical weights and native races.',
        adaptation='No optimizer or label fitting. One TRAIN scene numerical backward on output square. No DEV/TEST model scoring, no semantic use of old 11-class head.',
        state_history='All 32 raw events and retained numerical state throughout the scene; drain before composition rounds. Independent scene reset only.',
        retained=['native temporal computation','separate keys/values','persistent indexed state','hard receipt/value races','local counterfactual teacher','pure emission timer','source-relative pose','all eight object levels/four stages'],
        missing=['learned query/ownership/emission policies','sparse content discovery','global event-order credit','complete arithmetic/traffic ledger'],
        restrictions='Geometric adjacent queries and chronological drain barriers are explicit reference-executor restrictions. Current full frontier is a capability construction, not measured async hardware.',
        resources='Existing AWS six-slot coordinator slot6, one pinned core/thread; RSS3000000KiB,VMS12000000KiB,8GiBMemAvailable reserve,watchdog,1200s timeout.',
        decision='Measure incremental candidate/update/message work and deep gradient support; select task-specific fitting design or write/emission decoupling. No training automatically follows.')
    path=ROOT/f'experiments/hierarchy/{TAG}_admission.json'
    with path.open('x') as f:json.dump(declaration,f,indent=2);f.write('\n')
    original=json.loads((ROOT/'experiments/results/hierarchy/aws_g_hier_native_frozen_readout_20261015_v1.json').read_text())
    names=[n for n in original['source_sha256'] if n.startswith('sleeping_machines/')]
    names+=['experiments/hierarchy/frontier_preflight.py','experiments/hierarchy/part_combination_task.py','experiments/hierarchy/native_composition_data.py','sleeping_machines/frontier_native_composition.py','sleeping_machines/variance_reduced_native_composition.py','sleeping_machines/pure_race_timer.py','tests/test_part_combination_task.py','tests/test_frontier_native_composition.py','experiments/theory/STABILIZED_COMPOSITION_FRONTIERS.md','scripts/admit_frontier_composition_preflight.py',str(path.relative_to(ROOT)),'experiments/results/hierarchy/aws_g_hier_native_composition_20261015_v1_learn.final.pt']
    q=ROOT/f'experiments/queue/hierarchy/{TAG}/{TAG}.txt';q.parent.mkdir(parents=True,exist_ok=False)
    q.write_text('# G-HIER: '+declaration['decision']+'\n'+TAG+' experiments/hierarchy/frontier_preflight.py --tag '+TAG+'\n')
    job=dict(name=TAG,kind='hierarchy',queue=str(q.relative_to(ROOT)),queue_sha256=sha(q),source_sha256={n:sha(ROOT/n) for n in sorted(set(names))},result=f'experiments/results/hierarchy/{TAG}.json',requires=[],timeout_s=1200,rss_kb=3000000,vms_kb=12000000)
    marker=PLAN/'addenda'/('000_'+TAG+'.json')
    with marker.open('x') as f:json.dump(dict(host='ip-172-31-47-132',battle='G-HIER',purpose=declaration['decision'],parent_manifest_sha256=sha(PLAN/'manifest.json'),slots={'6':[job]}),f,indent=2);f.write('\n')
    print(marker.relative_to(ROOT))
if __name__=='__main__':main()
