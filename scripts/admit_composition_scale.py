"""One fixed-architecture broader-data continuation;gate before admission."""
import argparse,hashlib,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.hierarchy.composition_scale_learning import SOURCES,AUDIT,PARENT,PARENT_RESULT,PARENT_FEATURES
from experiments.hierarchy.native_composition_data import source_hashes
PLAN=ROOT/'experiments/queue/aws_model_improvement_repair_20261005T161000Z'
TAG='aws_g_hier_composition_scale_20261016_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--fit',action='store_true');a=p.parse_args()
 assert os.uname().nodename=='ip-172-31-47-132'
 active=json.loads((PLAN/'worker_recovery.aws6.status.json').read_text())['active'];assert '3' not in active
 assert int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))>8*1024*1024
 declaration=dict(battle='G-HIER',task_stationarity='Independent complete static scenes;not an online/cross-scene memory claim.',fighter='Fixed newest integrated identity-role native actor,continued from exact completed64update model/Adam/RNG. Tests adequacy of data and optimization before architectural substitution. Original acquired0.686609 whole-label reference retained.',retained='8abstraction levels/32native stages;temporal transport,hard native receipt/timer races,counterfactual receipt credit,separate keys/values,addressed persistent state,generated views,immutable part bindings,identity-preserving role relations and parent-used local teaching.',missing='Static observed ownership and query/arming explicit;all current participant/pair relation work paid,not sparse partner discovery;recursive likelihood heads inactive. Fresh data does not enlarge finite64TRAIN topology set.',adaptation_state='TRAIN-only updates,AdamWlr.001/wd.01,wholeCE+.25issued-view identityCE unchanged.2048additional paired updates/4096presentations;two without-replacement epochs over1024pairs/2048unique scenes. Persistent state within each scene,reset across independent scenes. Exact checkpoints every8updates.',data='Original64TRAIN arrangements under32new timing/order/translation/relative-spacing tapes1000..1031. No old DEV/TEST arrangements admitted. No hidden identities/arrangement bits supplied to actor;shape vocabulary reused from original TRAIN.',evaluation='Original TRAIN64 ridge.01 readout on original reused DEV64 is primary comparable endpoint. TRAIN128 broader-context readout separate,with matched initial score. Intermediate DEV at128/512/2048;fresh DEV64 tapes2000/2001 accessed only final. TEST never scored. Fresh DEV reuses abstract held-out arrangements and is not a new public benchmark.',decision='Does adequate exposure/training recover useful held-out composition with the fixed actor? Final legacy DEV NLL below saved0.686609 is within-family improvement;strong milestone requires>=80%accuracy and NLL<0.5 on old/fresh DEV. No automatic further fitting or new regularization/architecture.',resources='Gym slot3 one thread,RSS3MKiB,VMS12MKiB,host8GiBreserve,watchdogs. Actual optimizer/recovery gate before fullfit;timeout22000s selected from measured paired work plus all evaluation. Charge parent acquisition separately as reused evidence;record new native/teacher/pair/optimizer time and evaluation work. No fullFLOP/hardware/iso-compute claim.')
 admission=ROOT/f'experiments/hierarchy/{TAG}_admission.json'
 if not admission.exists():admission.write_text(json.dumps(declaration,indent=2)+'\n')
 else:assert json.loads(admission.read_text())==declaration
 sources=source_hashes();sources.update({n:sha(ROOT/n) for n in SOURCES+['experiments/hierarchy/composition_scale_learning.py','experiments/hierarchy/composition_scaling_data.py','tests/test_composition_scaling_data.py','tests/test_composition_scale_schedule.py']});sources.update({str(p.relative_to(ROOT)):sha(p) for p in [PARENT,PARENT_RESULT,PARENT_FEATURES,ROOT/AUDIT,admission]})
 gate=TAG+'_gate';name=TAG if a.fit else gate
 if a.fit:
  result=json.loads((ROOT/f'experiments/results/hierarchy/{gate}.json').read_text());assert result['gate']=='PASS' and result['exact_optimizer_resume']
  assert all(sha(ROOT/n)==h for n,h in result['source_sha256'].items())
 q=ROOT/f'experiments/queue/hierarchy/{name}/{name}.txt';q.parent.mkdir(parents=True,exist_ok=True);q.write_text('# G-HIER fixed-architecture data/training scaling;no ablation/control or TEST.\n'+name+' experiments/hierarchy/composition_scale_learning.py --tag '+name+' --mode '+('fit' if a.fit else 'gate')+'\n')
 job=dict(name=name,kind='hierarchy',queue=str(q.relative_to(ROOT)),queue_sha256=sha(q),source_sha256=sources,result=f'experiments/results/hierarchy/{name}.json',requires=[dict(result=f'experiments/results/hierarchy/{gate}.json',stage='composition_scale_gate')] if a.fit else [],timeout_s=22000 if a.fit else 240,rss_kb=3000000,vms_kb=12000000)
 output=PLAN/'addenda'/('000_'+name+'.json');assert not output.exists();packet=dict(host=os.uname().nodename,battle='G-HIER',purpose=declaration,parent_manifest_sha256=sha(PLAN/'manifest.json'),slots={'3':[job]});tmp=output.with_suffix('.writing');tmp.write_text(json.dumps(packet,indent=2)+'\n');tmp.replace(output);print('Admitted '+name,flush=True)
if __name__=='__main__':main()
