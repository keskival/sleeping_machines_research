"""Admit the product-ordered eight-level learned-decay RMR scaling job."""
import hashlib,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];PLAN=ROOT/'experiments/queue/aws_model_improvement_repair_20261005T161000Z'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 assert os.uname().nodename=='ip-172-31-47-132'
 gate='experiments/results/diagnostics/curie_rmr_contracts_20261017_v3.json'
 evidence=json.loads((ROOT/gate).read_text());assert evidence['status']=='passed'
 for name,digest in evidence['source_sha256'].items():assert sha(ROOT/name)==digest,name
 tag='aws_r1_rmr_residdecay_K8_16M_s0_20261017_v1';q=ROOT/f'experiments/queue/r1/{tag}/{tag}.txt';assert q.exists()
 sources=['experiments/r1_rmr.py','experiments/r1_cerl.py','experiments/r1_deep_keyed_lm.py','experiments/r1_token_keyed_lm.py',gate]
 purpose=dict(battle='R1',stationarity='Stationary one-pass 16M FineWeb GPT-2 development protocol. This fixed-weight checkpoint score is separate from non-stationary online evaluation.',fighter='Product-ordered eight-level residual memory with learned decay of old-slot corrections; the demonstrated four-level old-slot construction achieved4.9901 flagship versus5.0634 CERL and5.2034 dense L16, but regressed on longer lanes. Curie independently runs the K4 repair.',online='Training adapts causally on scored lane batches. Official development scoring freezes weights. Not an online learning win; the persistent event-local learner is evaluated separately.',state_history='Original256-token training lanes and prescribed1024/8192-token frozen evaluation history. No shortened evaluation. Temporal state resets per existing stationary lane protocol; carried cross-lane memory is not implemented in this driver.',retained='Native cause races, addressed episodic pointers, separate keys/values, dormant concepts, counterfactual exploration and current/old-slot residual memory. Carrier-backed temporal state; dense cross-slot reads. Independent asynchronous event stack, sparse read discovery, standing-slot pooling and cross-lane persistence are missing and explicitly not claimed.',restrictions='Depth8,old-slot top_m4,read_topk0,learned old_decay initialized0.01,16Mtokens,seed0 from product order. Same data and canonical scoring as saved references,unequal parameters/work. No new dense comparator fit or parameter sweep.',accounting='Existing AWS gym slot5,one thread,3MKiB RSS/12MKiB VMS,8GiB MemAvailable reserve,RSSwatchdog,24h timeout,checkpoint every256updates. Current sparse-address old-write algebra avoids dense T-by-T write storage. Driver FLOPs are operator estimates,not complete arithmetic/special-function/traffic or hardware measurements.',decision='Does learned decay repair long-history correction accumulation while eight levels improve flagship and selection accuracy at16M? Report all endpoints,including the65528target long-lane slice,without promoting only the favorable window. Reuse saved dense and native comparisons.',gate=dict(result=gate,sha256=sha(ROOT/gate),loaded_sources_match=True))
 job=dict(name=tag,kind='r1',queue=str(q.relative_to(ROOT)),queue_sha256=sha(q),source_sha256={n:sha(ROOT/n) for n in sources},result=f'experiments/results/token_language/{tag}.json',requires=[],timeout_s=86400,rss_kb=3000000,vms_kb=12000000)
 packet=dict(host=os.uname().nodename,battle='R1',purpose=purpose,parent_manifest_sha256=sha(PLAN/'manifest.json'),slots={'5':[job]})
 path=PLAN/'addenda/000_aws_r1_residual_decay_depth_20261017_v1.json'
 if path.exists():assert json.loads(path.read_text())==packet
 else:
  tmp=path.with_suffix('.writing');tmp.write_text(json.dumps(packet,indent=2)+'\n');tmp.replace(path)
 print(path)
if __name__=='__main__':main()
