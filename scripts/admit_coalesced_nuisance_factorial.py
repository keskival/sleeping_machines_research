"""Admit one frozen causal nuisance audit, no refit or new benchmark scoring."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];PLAN=ROOT/'experiments/queue/aws_model_improvement_repair_20261005T161000Z'
TAG='aws_g_hier_nuisance_factorial_20261015_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    result='experiments/results/hierarchy/aws_g_hier_coalesced_readout_20261015_v1.json'
    evidence=json.loads((ROOT/result).read_text());assert evidence['status']=='completed'
    declaration=dict(battle='G-HIER',stationarity='Stationary TRAIN scene interventions; no online claim.',
      strongest_fighter='Same acquired initial/final32-native-stage coalesced cores, persistent indexed memory,keys/values,native receipt and emission races unchanged.',
      adaptation='None. Four TRAIN pairs,two root orientations,coordinate/timing factorial. No DEV/TEST model scoring.',
      state='All32 raw events and native state throughout each independent scene; no truncation. One shared noise tape includes native stochastic-schedule sensitivity.',
      restrictions='Explicit phase barriers/geometric ownership; no learned policy or global order-boundary teacher. No hidden IDs in model input.',
      resources='AWSslot6,one pinned CPU/thread,RSS3MKiB,VMS12MKiB,8GiBreserve,watchdogs,900s; cache features.',
      decision='Disentangle measured coordinate versus timing/order sensitivity before architecture repair; no new fit.')
    path=ROOT/f'experiments/hierarchy/{TAG}_admission.json';path.write_text(json.dumps(declaration,indent=2)+'\n')
    names=list(evidence['source_sha256'])+['experiments/hierarchy/coalesced_nuisance_factorial.py','tests/test_coalesced_nuisance_factorial.py','scripts/admit_coalesced_nuisance_factorial.py',str(path.relative_to(ROOT)),result]
    q=ROOT/f'experiments/queue/hierarchy/{TAG}/{TAG}.txt';q.parent.mkdir(parents=True,exist_ok=False)
    q.write_text('# G-HIER: '+declaration['decision']+'\n'+TAG+' experiments/hierarchy/coalesced_nuisance_factorial.py --tag '+TAG+'\n')
    job=dict(name=TAG,kind='hierarchy',queue=str(q.relative_to(ROOT)),queue_sha256=sha(q),source_sha256={n:sha(ROOT/n) for n in sorted(set(names))},result=f'experiments/results/hierarchy/{TAG}.json',requires=[dict(result=result,stage='coalesced_frozen_readout')],timeout_s=900,rss_kb=3000000,vms_kb=12000000)
    marker=PLAN/'addenda'/('000_'+TAG+'.json')
    with marker.open('x') as f:json.dump(dict(host='ip-172-31-47-132',battle='G-HIER',purpose=declaration['decision'],parent_manifest_sha256=sha(PLAN/'manifest.json'),slots={'6':[job]}),f,indent=2);f.write('\n')
    print(marker.relative_to(ROOT))
if __name__=='__main__':main()
