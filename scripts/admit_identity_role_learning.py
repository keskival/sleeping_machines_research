"""Source-bound integrated identity/role repair;gate before one acquisition."""
import argparse,hashlib,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.hierarchy.identity_role_learning import SOURCES,AUDIT,PARENT
from experiments.hierarchy.native_composition_data import source_hashes
PLAN=ROOT/'experiments/queue/aws_model_improvement_repair_20261005T161000Z'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--fit',action='store_true');a=p.parse_args()
 assert os.uname().nodename=='ip-172-31-47-132'
 state=json.loads((PLAN/'worker_recovery.aws6.status.json').read_text());assert '3' not in state['active']
 assert int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))>8*1024*1024
 tag='aws_g_hier_identity_role_20261016_v1';gate=tag+'_gate';name=tag if a.fit else gate
 admission=ROOT/f'experiments/hierarchy/{tag}_admission.json'
 declaration=dict(battle='G-HIER',task_stationarity='Independent static revealed scenes;not an online benchmark.',fighter='Latest learned native part-identity actor with measured12.5%→50.2% DEV shape gain. Repair coarse-role overwrite using separately addressed participants and teach parent-used immutable versions at their issue times. Original acquired0.686609 whole-label reference retained.',adaptation_state='TRAIN whole-label and observed shape supervision;state persists within scenes,reset across independent scenes.64paired updates,128presentations;TRAIN-only ridge.01head,reusedDEV,TEST closed.',retained='8abstraction levels/32native stages,computational delays,hard native receipt and timer races,counterfactual receipt learning,separate keys/values,addressed persistent memory,generated views,immutable version directories.',missing='Explicit static ownership/query/arming. Relation operation currently reads all retained children and all pairs;quadratic work charged,not sparse partner selection. Recursive likelihood heads inactive. No semantic taxonomy or general streaming ownership claim.',objective='Whole CE+fixed0.25mean observed identity CE on actual parent-used versions. Shape targets only,no hidden identities or target inputs. This endpoint differs from prior latest-memory identity score;do not compare their accuracies as matched endpoints.',resources='One pinned gym slot3,RSS3MKiB,VMS12MKiB,host reserve8GiB,watchdogs,exact model/Adam/RNG/source/input/order checkpoint every8updates. Numerical-state inference checkpoints rekey in-flight envelopes;training credit resumes only at scene/update boundaries. All relation reads/pairs/maps charged separately;no full arithmetic/hardware claim.',decision='Does the identity-preserving temporal interface improve whole-label DEV versus initial candidate and saved acquired0.686609? Report local issued-version accuracy separately. No control/ablation acquisition,no automatic longer scale-up.')
 if not admission.exists():admission.write_text(json.dumps(declaration,indent=2)+'\n')
 else:assert json.loads(admission.read_text())==declaration
 sources=source_hashes();sources.update({n:sha(ROOT/n) for n in SOURCES});sources.update({AUDIT:sha(ROOT/AUDIT),str(PARENT.relative_to(ROOT)):sha(PARENT),str(admission.relative_to(ROOT)):sha(admission)})
 if a.fit:
  g=json.loads((ROOT/f'experiments/results/hierarchy/{gate}.json').read_text());assert g['gate']=='PASS' and g['stage']=='identity_role_gate'
  assert all(sha(ROOT/n)==v for n,v in g['source_sha256'].items())
 q=ROOT/f'experiments/queue/hierarchy/{name}/{name}.txt';q.parent.mkdir(parents=True,exist_ok=True);q.write_text('# G-HIER integrated identity-preserving role repair;no ablations,TRAIN only.\n'+name+' experiments/hierarchy/identity_role_learning.py --tag '+name+' --mode '+('fit' if a.fit else 'gate')+' --steps 64\n')
 job=dict(name=name,kind='hierarchy',queue=str(q.relative_to(ROOT)),queue_sha256=sha(q),source_sha256=sources,result=f'experiments/results/hierarchy/{name}.json',requires=[dict(result=f'experiments/results/hierarchy/{gate}.json',stage='identity_role_gate')] if a.fit else [],timeout_s=1800 if a.fit else 240,rss_kb=3000000,vms_kb=12000000)
 target=PLAN/'addenda'/('000_'+name+'.json');assert not target.exists();packet=dict(host=os.uname().nodename,battle='G-HIER',purpose=declaration,parent_manifest_sha256=sha(PLAN/'manifest.json'),slots={'3':[job]});tmp=target.with_suffix('.writing');tmp.write_text(json.dumps(packet,indent=2)+'\n');tmp.replace(target)
 print('Admitted '+name+';one integrated candidate,no ablation arms.')
if __name__=='__main__':main()
