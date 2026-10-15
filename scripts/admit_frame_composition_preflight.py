"""One acquired-weight observed-frame contract/resource gate, no fitting."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];PLAN=ROOT/'experiments/queue/aws_model_improvement_repair_20261005T161000Z'
TAG='aws_g_hier_frame_preflight_20261015_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    predecessor='experiments/results/hierarchy/aws_g_hier_nuisance_factorial_20261015_v1.json'
    evidence=json.loads((ROOT/predecessor).read_text());assert evidence['status']=='completed'
    declaration=dict(battle='G-HIER',stationarity='Independent stationary TRAIN scenes, numerical mechanism gate only.',
      strongest_fighter='32native-stage/eight-level generated-view compositor with persistent world-pixel memories and upper observed-frame ownership. Same acquired parameters/shapes; no new primitive replaces races.',
      adaptation='None.32factorial TRAIN scenes,one32-event backward on output square. No label fitting or DEV/TEST model scoring.',
      state_history='Every raw event updates persistent world-pixel memory. Establish frame only after revealed queue drains, no sensor replay; commit frame for episode. Numerical memory retained independently of credit.',
      retained=['native receipt/value races','separate keys/values','persistent indexed state','local counterfactual credit','native generated emissions','pure timer interior credit','relative pose','all eight levels/four native stages'],
      restrictions='Observed scene frame and drain barriers are explicit, not learned ownership/local streaming policy. Root support128pixels. No rotation/timing invariance assumed; no exact event-order teacher.',
      resources='AWSslot6,one pinned CPU/thread,RSS3MKiB,VMS12MKiB,8GiBreserve,watchdogs,900s. Charge extra world-pixel state and frame operations.',
      decision='Verify removal of measured translation sensitivity and check timing/gradient/resource consequences before any new fit.')
    path=ROOT/f'experiments/hierarchy/{TAG}_admission.json'
    with path.open('x') as f:json.dump(declaration,f,indent=2);f.write('\n')
    names=list(evidence['source_sha256'])+['experiments/hierarchy/frame_preflight.py','sleeping_machines/frame_native_composition.py','tests/test_frame_native_composition.py','experiments/theory/OBSERVED_FRAME_NATIVE_COMPOSITION.md','scripts/admit_frame_composition_preflight.py',str(path.relative_to(ROOT)),predecessor]
    q=ROOT/f'experiments/queue/hierarchy/{TAG}/{TAG}.txt';q.parent.mkdir(parents=True,exist_ok=False)
    q.write_text('# G-HIER: '+declaration['decision']+'\n'+TAG+' experiments/hierarchy/frame_preflight.py --tag '+TAG+'\n')
    job=dict(name=TAG,kind='hierarchy',queue=str(q.relative_to(ROOT)),queue_sha256=sha(q),source_sha256={n:sha(ROOT/n) for n in sorted(set(names))},result=f'experiments/results/hierarchy/{TAG}.json',requires=[dict(result=predecessor,stage='coalesced_nuisance_factorial')],timeout_s=900,rss_kb=3000000,vms_kb=12000000)
    marker=PLAN/'addenda'/('000_'+TAG+'.json')
    with marker.open('x') as f:json.dump(dict(host='ip-172-31-47-132',battle='G-HIER',purpose=declaration['decision'],parent_manifest_sha256=sha(PLAN/'manifest.json'),slots={'6':[job]}),f,indent=2);f.write('\n')
    print(marker.relative_to(ROOT))
if __name__=='__main__':main()
