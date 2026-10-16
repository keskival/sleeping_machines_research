"""Admit the fully native deep language/local-credit TRAIN gate only."""
import argparse,hashlib,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'experiments')]
import experiments.hierarchy.local_native_language_gate
from experiments.hierarchy.native_composition_data import source_hashes
PLAN=ROOT/'experiments/queue/aws_model_improvement_repair_20261005T161000Z'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--tag',default='aws_r1_local_native_deep_gate_20261016_v2');a=ap.parse_args()
 assert os.uname().nodename=='ip-172-31-47-132'
 active=json.loads((PLAN/'worker_recovery.aws6.status.json').read_text())['active'];assert '4' not in active
 assert int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))>8*1024*1024
 name=a.tag
 declaration=dict(battle='R1',task_stationarity='Fixed stationary FineWeb TRAIN numerical gate;not a competitive or online comparison.',fighter='Eight-layer native race/receiver/transport core with local own-write eligibility;normalized repeat/continue/concept/background and depth cause races. Zero SSM/attention/MLP carrier layers.',state_policy='Persistent receiver numerical state and local eligibility;no recurrent BPTT. Only32TRAIN targets in one gate;no DEV/TEST. Current head episodic history spans this supplied credit lane;cross-credit indexed history is a missing integration and blocks flagship acquisition.',online_adaptation='One optimizer step after scoring TRAIN forecasts;asynchronous receiver optimizer/versioning not yet integrated. No online-win claim.',retained='Computational delays,independent native head races,sparse selected receiver writes,temporal transport,separate keys/values,one sampled losing proposal,receiver-local past-write eligibility,dormant concepts with exact product-key top-k and paid exploration.',restrictions='Local own-state derivative conditional on input/routing/timestamps;changed-parameter traces approximate. Higher-layer incoming gradients stop at local message boundaries;every depth receives direct cause-posterior credit. All core pool keys and their local trace projections are paid. Gate only,not a restricted family verdict.',resource_accounting='Slot4 one thread,3MKiBRSS/12MKiBVMS,8GiB host reserve,600s timeout;actual32target optimizer time,per-layer credit,scoredkeys,winner/alternativework,eligibility state,headwork,RSS and source hashes. No full FLOP/hardware claim.',decision='Does the fully native deep model support finite local learning on real tokenized data,and what bottleneck must be repaired before an acquisition against the saved dense hybrid? No automatic fit.')
 sources=source_hashes()
 for n in ['experiments/r1_cerl.py','experiments/r1_local_native_deep.py','experiments/r1_deep_keyed_lm.py','experiments/hierarchy/local_native_language_gate.py','tests/test_native_receiver_eligibility.py','tests/test_local_deep_race.py','tests/test_local_native_deep_lm.py']:sources[n]=sha(ROOT/n)
 q=ROOT/f'experiments/queue/r1/{name}/{name}.txt';q.parent.mkdir(parents=True,exist_ok=True);assert not q.exists();q.write_text('# R1 fully native deep local-credit gate;no competitive fit.\n'+name+' experiments/hierarchy/local_native_language_gate.py --tag '+name+'\n')
 job=dict(name=name,kind='r1',queue=str(q.relative_to(ROOT)),queue_sha256=sha(q),source_sha256=sources,result=f'experiments/results/diagnostics/{name}.json',requires=[],timeout_s=600,rss_kb=3000000,vms_kb=12000000)
 target=PLAN/'addenda'/('000_'+name+'.json');assert not target.exists();packet=dict(host=os.uname().nodename,battle='R1',purpose=declaration,parent_manifest_sha256=sha(PLAN/'manifest.json'),slots={'4':[job]});tmp=target.with_suffix('.writing');tmp.write_text(json.dumps(packet,indent=2)+'\n');tmp.replace(target);print('Admitted fully native deep language/local-credit gate;no acquisition.')
if __name__=='__main__':main()
