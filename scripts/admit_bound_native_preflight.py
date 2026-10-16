"""One native hierarchy integration gate through the existing gym coordinator."""
import hashlib,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];PLAN=ROOT/'experiments/queue/aws_model_improvement_repair_20261005T161000Z'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 assert os.uname().nodename=='ip-172-31-47-132'
 tag='aws_g_hier_bound_native_preflight_20261016_v1'
 # Only imports,not numerical execution;all model work occurs under run_safe.
 import sys;sys.path.insert(0,str(ROOT))
 from experiments.hierarchy.bound_native_preflight import SOURCES,PARENT
 from experiments.hierarchy.native_composition_data import source_hashes
 hashes=source_hashes();hashes.update({n:sha(ROOT/n) for n in SOURCES});hashes[str(PARENT.relative_to(ROOT))]=sha(PARENT)
 declaration=dict(battle='G-HIER',task_stationarity='Independent stationary TRAIN scenes;local future mark is a connectivity contract only.',strongest_fighter='Strongest acquired original relational native actor;geometry/carrier extensions failed to improve its class transfer,so reuse its weights unchanged.',state_history='Eight persistent abstraction levels and32native stages,no within-scene reset. New versioned binding directories preserve old child snapshots;explicit GC keeps referenced/current/inflight versions.',online_adaptation='No acquisition. One actual CE backward and a separate local-race backward;no optimizer or target-driven state/weight update.',retained_mechanisms='Native temporal computation,hard receipt races,counterfactual credit,generated timer views,persistent indexed state,separate keys/values,existing role interactions. Adds lower-memory version access and native competing-risk prediction from the full declared bound pool.',restrictions='Observed spatial ownership/frame and explicit query/arming policy remain;no learned semantic identities or autonomous learned ownership. Arbitrary local mark has no quality interpretation. No TEST access.',resources='Slot3,one pinned physical CPU core,RSS3MKiB/VMS12MKiB,8GiB host reserve,watchdogs;all extra reads/retained records/metadata and local head work disclosed.',decision='Does binding-directory integration preserve the acquired actor exactly and support live local race credit across real multi-memory32stage composition within the resource envelope?')
 admission=ROOT/f'experiments/hierarchy/{tag}_admission.json';admission.write_text(json.dumps(declaration,indent=2)+'\n')
 q=ROOT/f'experiments/queue/hierarchy/{tag}/{tag}.txt';q.parent.mkdir(parents=True,exist_ok=True)
 q.write_text('# G-HIER: acquired actor parity and multi-memory native race connectivity,TRAIN only;no fitting/TEST.\n'+tag+' experiments/hierarchy/bound_native_preflight.py --tag '+tag+'\n')
 hashes[str(admission.relative_to(ROOT))]=sha(admission)
 job=dict(name=tag,kind='hierarchy',queue=str(q.relative_to(ROOT)),queue_sha256=sha(q),source_sha256=hashes,result=f'experiments/results/hierarchy/{tag}.json',requires=[],timeout_s=900,rss_kb=3000000,vms_kb=12000000)
 marker=PLAN/'addenda'/('000_'+tag+'.json');packet=dict(host=os.uname().nodename,battle='G-HIER',purpose=declaration,parent_manifest_sha256=sha(PLAN/'manifest.json'),slots={'3':[job]})
 assert not marker.exists();tmp=marker.with_suffix('.writing');tmp.write_text(json.dumps(packet,indent=2)+'\n');tmp.replace(marker)
 print('Admitted acquired8level/32stage binding and race-pool gate in slot3.')
if __name__=='__main__':main()
