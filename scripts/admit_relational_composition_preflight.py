"""One real-size paired numerical/update gate for native role composition."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];PLAN=ROOT/'experiments/queue/aws_model_improvement_repair_20261005T161000Z'
TAG='aws_g_hier_relational_preflight_20261015_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    evidence=json.loads((ROOT/'experiments/results/hierarchy/aws_g_hier_frame_preflight_20261015_v1.json').read_text())
    declaration=dict(battle='G-HIER',task_stationarity='Stationary independent TRAIN scenes; numerical/resource gate,not an online benchmark.',
      strongest_fighter='Native frame actor plus persistent source-level/role view caches, native temporal transport and nonzero ordered bilinear child interactions;all32native stages retained. Paired native frame arm separates learning-pipeline repair.',
      adaptation='One paired AdamW update per native arm. Counterparts derived from TRAIN sensor arrays/existing whole labels only; no hidden tree IDs,DEV or TEST. Same random tape across pair preserves marginal race laws.',
      state_history='All32observations retained through scene-local memory and composition. Role cache bounded by source level/quadrant,latest view plus time. Numerical state is independent of gradient boundaries.',
      retained=['native hard temporal races','persistent addressed state','separate keys/values','local counterfactual teacher','native generated emissions','source-relative pose','protected/rotating temporal state','source-level-separated role caches'],
      restrictions='Observed-frame/geometric ownership and phase barriers;not learned object discovery. Nonzero extra operators change predictions and parameter/work counts. No exact future queue-order teacher.',
      resources='AWScoordinator slot6,one pinned core/thread,RSS3000000KiB,VMS12000000KiB,8GiBreserve,watchdogs,1800s.',
      decision='Measure full paired optimizer resource/deep/interaction gradients before64paired-update native control/operator comparison. No quality verdict from one update.')
    admission=ROOT/f'experiments/hierarchy/{TAG}_admission.json'
    with admission.open('x') as f:json.dump(declaration,f,indent=2);f.write('\n')
    names=list(evidence['source_sha256'])+['sleeping_machines/relational_frame_composition.py','experiments/hierarchy/observed_training_pairs.py','experiments/hierarchy/paired_composition_learning.py','experiments/hierarchy/relational_preflight.py','tests/test_relational_frame_composition.py','tests/test_observed_training_pairs.py','experiments/theory/ROLE_MEMORY_AND_INTERACTION_DEPTH.md','scripts/admit_relational_composition_preflight.py',str(admission.relative_to(ROOT))]
    q=ROOT/f'experiments/queue/hierarchy/{TAG}/{TAG}.txt';q.parent.mkdir(parents=True,exist_ok=False)
    q.write_text('# G-HIER: '+declaration['decision']+'\n'+TAG+' experiments/hierarchy/relational_preflight.py --tag '+TAG+'\n')
    job=dict(name=TAG,kind='hierarchy',queue=str(q.relative_to(ROOT)),queue_sha256=sha(q),source_sha256={n:sha(ROOT/n) for n in sorted(set(names))},result=f'experiments/results/hierarchy/{TAG}.json',requires=[],timeout_s=1800,rss_kb=3000000,vms_kb=12000000)
    marker=PLAN/'addenda'/('000_'+TAG+'.json')
    with marker.open('x') as f:json.dump(dict(host='ip-172-31-47-132',battle='G-HIER',purpose=declaration['decision'],parent_manifest_sha256=sha(PLAN/'manifest.json'),slots={'6':[job]}),f,indent=2);f.write('\n')
    print(marker.relative_to(ROOT))
if __name__=='__main__':main()
