"""Admit Curie's ordered race program through the existing six-slot coordinator."""
import hashlib,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];PLAN=ROOT/'experiments/queue/aws_model_improvement_repair_20261005T161000Z'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 assert os.uname().nodename=='ip-172-31-47-132'
 assert int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))>8*1024*1024
 sources=['experiments/hierarchy/curie_race_admission_gate.py','experiments/r1_cerl.py','experiments/r1_rmr.py','experiments/r1_deep_keyed_lm.py','experiments/r1_token_keyed_lm.py']
 gates={m:f'aws_r1_curie_{m}_admission_20261016_v1' for m in ['cerl','rmr']}
 complete=all((ROOT/f'experiments/results/diagnostics/{t}.json').exists() for t in gates.values())
 slots={}
 if not complete:
  for slot,m in [(1,'cerl'),(3,'rmr')]:
   tag=gates[m];q=ROOT/f'experiments/queue/r1/{tag}/{tag}.txt';q.parent.mkdir(parents=True,exist_ok=True);text=f'# Battle R1. Validate current merged source normalization, causality, exact zero-refinement recovery and memory algebra before ordered fits. No acquisition or TEST. One thread, six-slot guards.\n{tag} experiments/hierarchy/curie_race_admission_gate.py --tag {tag} --model {m}\n'
   if q.exists():assert q.read_text()==text
   else:q.write_text(text)
   slots[str(slot)]=[dict(name=tag,kind='r1',queue=str(q.relative_to(ROOT)),queue_sha256=sha(q),source_sha256={s:sha(ROOT/s) for s in sources},result=f'experiments/results/diagnostics/{tag}.json',requires=[],timeout_s=1800,rss_kb=3000000,vms_kb=12000000)]
  name='000_aws_r1_curie_current_gates_20261016_v1.json'
 else:
  for m,t in gates.items():
   r=json.loads((ROOT/f'experiments/results/diagnostics/{t}.json').read_text());assert r['gate']=='PASS' and r['status']=='completed';assert all(sha(ROOT/s)==h for s,h in r['source_sha256'].items())
  jobs=[(1,'cerl_16M'),(3,'rmr_dense_K2_4M'),(5,'rmr_dense_K8_4M'),(6,'cerl_pk65536_4M'),(2,'cerl_pk262144_4M'),(6,'cerl_msg_4M')]
  for slot,suffix in jobs:
   tag=f'aws_r1_{suffix}_s0_20261016_v1';q=ROOT/f'experiments/queue/r1/{tag}/{tag}.txt';assert q.exists();m='rmr' if suffix.startswith('rmr') else 'cerl'
   slots.setdefault(str(slot),[]).append(dict(name=tag,kind='r1',queue=str(q.relative_to(ROOT)),queue_sha256=sha(q),source_sha256={s:sha(ROOT/s) for s in sources},result=f'experiments/results/token_language/{tag}.json',requires=[dict(result=f'experiments/results/diagnostics/{gates[m]}.json',stage='current Curie numerical admission')],timeout_s=100000,rss_kb=3000000,vms_kb=12000000))
  name='000_aws_r1_curie_ordered_program_20261016_v1.json'
 packet=dict(host=os.uname().nodename,battle='R1',purpose='Ordered in PRODUCT_ORDERS: race prediction, residual-memory depth, capacity beyond activity and causal messages. Full per-arm declarations in queues; stationary fixed-weight development, shuffled lanes and lane-bounded memory, temporal-memory carrier and BPTT explicitly retained. These are race-model development runs, not the fully native event-local flagship. No external model training, TEST, or hardware latency claim. Preserve active jobs; six distinct cores, 8GiB reserve, watchdogs.',parent_manifest_sha256=sha(PLAN/'manifest.json'),slots=slots)
 path=PLAN/'addenda'/name
 if path.exists():assert json.loads(path.read_text())==packet
 else:
  tmp=path.with_suffix('.writing');tmp.write_text(json.dumps(packet,indent=2)+'\n');tmp.replace(path)
 print(name)
if __name__=='__main__':main()
