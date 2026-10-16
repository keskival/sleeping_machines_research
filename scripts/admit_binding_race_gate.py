"""Admit one integrated native local-binding contract, without acquisition."""
import hashlib,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.hierarchy.binding_race_gate import CHECKPOINTS,AUDIT,source_hashes
PLAN=ROOT/'experiments/queue/aws_model_improvement_repair_20261005T161000Z'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 assert os.uname().nodename=='ip-172-31-47-132'
 active=json.loads((PLAN/'worker_recovery.aws6.status.json').read_text())['active'];assert '4' not in active
 assert int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))>8*1024*1024
 name='aws_g_hier_binding_race_gate_20261016_v1'
 declaration=dict(battle='G-HIER',task_stationarity='One independent static TRAIN scene;no online comparison.',fighter='Frozen512update complete native identity-role actor plus external training-only binding race teacher.',state_policy='Persistent native numerical state within the independent scene;actual immutable parent publication and child issue views. No optimizer step,DEV or TEST.',retained='All8levels/32native stages,temporal computation,receipt/timer races,counterfactual credit,separate keys/values,versioned indexed memory and sparse addressed native updates.',restrictions='Latest published parents at levels2and above;all distinguishable single-pair content-to-pose swaps enumerated. Lower teacher views detached;whole-task credit unmodified. This is a numerical/resource gate,not a predictive model verdict or ablation.',resource_accounting='Actual native work plus bound reads,teacher candidate clocks,teacher parameters,backward credit,wall time and peakRSS. Slot4 one thread,3MKiBRSS/12MKiBVMS,8GiB host reserve,180second timeout. No full arithmetic/hardware claim.',decision='Can a parent receive finite local binding-race credit on actual issued memory views without changing native predictions? Gate a candidate capability before considering any acquisition after the fixed2048update result.')
 sources=source_hashes()
 for n in ['experiments/hierarchy/binding_race_gate.py','sleeping_machines/binding_race_teacher.py','tests/test_binding_race_teacher.py',AUDIT,CHECKPOINTS[512]]:sources[n]=sha(ROOT/n)
 q=ROOT/f'experiments/queue/hierarchy/{name}/{name}.txt';q.parent.mkdir(parents=True,exist_ok=True);assert not q.exists();q.write_text('# G-HIER actual TRAIN local-binding gate;no optimizer update.\n'+name+' experiments/hierarchy/binding_race_gate.py --tag '+name+'\n')
 job=dict(name=name,kind='hierarchy',queue=str(q.relative_to(ROOT)),queue_sha256=sha(q),source_sha256=sources,result=f'experiments/results/hierarchy/{name}.json',requires=[],timeout_s=180,rss_kb=3000000,vms_kb=12000000)
 target=PLAN/'addenda'/('000_'+name+'.json');assert not target.exists();packet=dict(host=os.uname().nodename,battle='G-HIER',purpose=declaration,parent_manifest_sha256=sha(PLAN/'manifest.json'),slots={'4':[job]});tmp=target.with_suffix('.writing');tmp.write_text(json.dumps(packet,indent=2)+'\n');tmp.replace(target);print('Admitted integrated binding-race numerical gate,slot4;no acquisition.')
if __name__=='__main__':main()
