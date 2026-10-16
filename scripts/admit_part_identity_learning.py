"""One integrated native identity acquisition,then stop;no new ablation arms."""
import hashlib,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.hierarchy.part_identity_learning import SOURCES,AUDIT,PARENT
from experiments.hierarchy.native_composition_data import source_hashes
PLAN=ROOT/'experiments/queue/aws_model_improvement_repair_20261005T161000Z'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 assert os.uname().nodename=='ip-172-31-47-132'
 state=json.loads((PLAN/'worker_recovery.aws6.status.json').read_text());assert '3' not in state['active']
 assert int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))>8*1024*1024
 tag='aws_g_hier_part_identity_20261016_v1';gate=tag+'_gate'
 declaration=dict(battle='G-HIER',task_stationarity='Independent revealed static scenes,not online-stream evaluation.',fighter='Newest integrated native complete-ownership/role-memory candidate,reusing strongest acquired original relational weights. Geometry/carrier variants did not improve its class transfer.',adaptation_state='Weights adapt on TRAIN whole labels and observable part shapes only. Persistent native state within each complete scene;reset across independent scenes.64paired updates,128presentations;TEST closed.',retained='8abstraction levels/32native stages,hard temporal receipt races,counterfactual credit,separate keys/values,sparse addressed state,native timer views,versioned part access,existing role interactions.',missing='Ownership/grouping and query/arming policy explicit,not learned. Recursive race likelihood heads available but inactive. No learned semantic taxonomy,continuous ownership revision or cross-scene persistent-state claim.',teacher='TRAIN-only canonical relative pixel/polarity signatures provide identity targets only;no hidden generator IDs,arrangement bits or inference teacher inputs.Whole CE+fixed0.25mean part CE;no sweep.',resources='Actual paired-update gate,then one pinned slot3 run;RSS3MKiB,VMS12MKiB,8GiB host reserve/watchdogs,exact8update checkpoints. Native/teacher work charged;no new dense training or ablation/control acquisition.',decision='Does complete native part ownership plus learned observable identity improve held-out combination development quality over initial candidate and saved acquired reference0.686609?Shape learning and whole-label transfer reported separately;no family verdict or automatic scale-up.')
 admission=ROOT/f'experiments/hierarchy/{tag}_admission.json';admission.write_text(json.dumps(declaration,indent=2)+'\n')
 sources=source_hashes();sources.update({n:sha(ROOT/n) for n in SOURCES});sources.update({AUDIT:sha(ROOT/AUDIT),str(PARENT.relative_to(ROOT)):sha(PARENT),str(admission.relative_to(ROOT)):sha(admission)})
 jobs=[]
 for name,mode in [(gate,'gate'),(tag,'fit')]:
  q=ROOT/f'experiments/queue/hierarchy/{name}/{name}.txt';q.parent.mkdir(parents=True,exist_ok=True);q.write_text('# G-HIER: one integrated native ownership+identity learner;TRAIN-observable targets,no ablation/control fit or TEST. See source-bound admission.\n'+name+' experiments/hierarchy/part_identity_learning.py --tag '+name+' --mode '+mode+' --steps 64\n')
  jobs.append(dict(name=name,kind='hierarchy',queue=str(q.relative_to(ROOT)),queue_sha256=sha(q),source_sha256=sources,result=f'experiments/results/hierarchy/{name}.json',requires=[] if mode=='gate' else [dict(result=f'experiments/results/hierarchy/{gate}.json',stage='part_identity_gate')],timeout_s=180 if mode=='gate' else 1800,rss_kb=3000000,vms_kb=12000000))
 p=PLAN/'addenda'/('000_'+tag+'.json');assert not p.exists();packet=dict(host=os.uname().nodename,battle='G-HIER',purpose=declaration,parent_manifest_sha256=sha(PLAN/'manifest.json'),slots={'3':jobs});tmp=p.with_suffix('.writing');tmp.write_text(json.dumps(packet,indent=2)+'\n');tmp.replace(p)
 print('Admitted actual paired gate→single64update native identity acquisition,slot3;no ablation arms.')
if __name__=='__main__':main()
