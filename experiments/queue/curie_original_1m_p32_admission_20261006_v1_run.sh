#!/usr/bin/env bash
set -euo pipefail
cd /workspace
for attempt in $(seq 1 50); do
  if ! tmux has-session -t curie_original_1m_utility_admission_20261006_v1 2>/dev/null; then break; fi
  sleep 30
done
if tmux has-session -t curie_original_1m_utility_admission_20261006_v1 2>/dev/null; then exit 75; fi
python3 - <<'CHECK'
import hashlib,json
from pathlib import Path
u=json.loads(Path('experiments/results/diagnostics/curie_original_1m_utility_20261006_v1.json').read_text())
assert u['status']=='completed' and u['rows'][0]['context_gain']>0
assert u['rows'][0]['partition_parity'] and u['rows'][0]['matched_rng']
r=json.loads(Path('experiments/results/token_language/curie_data_growth_tokens_256k_b64_c16_p32_s6_20261005_v1.json').read_text())
assert r['status']=='completed' and r['presentations_total']==524272
v=json.loads(Path('experiments/results/diagnostics/curie_data_growth_tokens_256k_p32_utility_20261005_v1.json').read_text())
assert v['status']=='completed' and v['rows'][0]['context_gain']>0
for name,digest in r['identity']['source_sha256'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
for key,arg in (('train_sha256','train_file'),('dev_sha256','dev_file')):
    h=hashlib.sha256()
    with open(r['args'][arg],'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    assert h.hexdigest()==r['identity'][key],key
CHECK
# Completed P32/256K1177s and P24/1M4376s motivate9000s allowance.
MEM_CAP_KB=5000000 MEM_CAP_RSS_KB=800000 MIN_AVAIL_MB=8192 JOB_TIMEOUT_S=9000 bash experiments/queue/run_safe.sh experiments/queue/curie_data_growth_tokens_1m_b64_c16_p32_s6_20261005_v1.txt
