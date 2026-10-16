"""One frozen actual-actor credit audit,not a new training/ablation arm."""
import hashlib,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.hierarchy.composition_credit_audit import CHECKPOINTS,AUDIT
from experiments.hierarchy.native_composition_data import source_hashes
PLAN=ROOT/'experiments/queue/aws_model_improvement_repair_20261005T161000Z'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 assert os.uname().nodename=='ip-172-31-47-132'
 active=json.loads((PLAN/'worker_recovery.aws6.status.json').read_text())['active'];assert '4' not in active
 assert int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))>8*1024*1024
 name='aws_g_hier_composition_credit_audit_20261016_v1'
 declaration=dict(battle='G-HIER',task_stationarity='Independent static TRAIN scenes;no online comparison.',fighter='Actual frozen128/512update integrated identity-role actor snapshots and actual joint head;not a proxy or changed model.',state_policy='Native persistent state within scenes;reset across independent scenes;fixed paired clock tape113. No parameter/optimizer changes;no DEV/TEST.',retained='Full actual8level32stage native actor,clocks,hard receipt/timer races,counterfactual teacher,key/value separation,versioned lower-memory access and role interactions.',restrictions='Eight original TRAIN pairs only;local linear-VJP/cancellation characterization,not whole-distribution attribution. Fixed snapshot numerical state and common randomness declare scope. Does not compare ablated models.',resource_accounting='All native/counterfactual forward work,two backward traversals per scene,peakRSS and wall time recorded. Slot4 one thread,3MKiBRSS/12MKiBVMS,8GiB host reserve,900second timeout. No full arithmetic/hardware claim.',decision='Does the actual whole-task teacher distinguish opposite-label representations,or does its signal cancel by level/operator? Identify a measured sensitivity/credit bottleneck while the fixed2048update acquisition finishes unchanged.')
 sources=source_hashes();sources.update({n:sha(ROOT/n) for n in ['experiments/hierarchy/composition_credit_audit.py','tests/test_composition_credit_algebra.py',AUDIT,*CHECKPOINTS.values()]})
 q=ROOT/f'experiments/queue/hierarchy/{name}/{name}.txt';q.parent.mkdir(parents=True,exist_ok=True);q.write_text('# G-HIER frozen TRAIN-only credit contract;no ablation/fit.\n'+name+' experiments/hierarchy/composition_credit_audit.py --tag '+name+'\n')
 job=dict(name=name,kind='hierarchy',queue=str(q.relative_to(ROOT)),queue_sha256=sha(q),source_sha256=sources,result=f'experiments/results/hierarchy/{name}.json',requires=[],timeout_s=900,rss_kb=3000000,vms_kb=12000000)
 target=PLAN/'addenda'/('000_'+name+'.json');assert not target.exists();packet=dict(host=os.uname().nodename,battle='G-HIER',purpose=declaration,parent_manifest_sha256=sha(PLAN/'manifest.json'),slots={'4':[job]});tmp=target.with_suffix('.writing');tmp.write_text(json.dumps(packet,indent=2)+'\n');tmp.replace(target);print('Admitted frozen native paired-credit audit,slot4;no training/ablation.')
if __name__=='__main__':main()
