#!/usr/bin/env bash
set -euo pipefail
cd /workspace
python3 - <<'CHECK'
import hashlib,json
from pathlib import Path
parent=Path('experiments/results/token_language/curie_data_growth_tokens_256k_b64_c16_p24_s6_20261005_v1.json')
r=json.loads(parent.read_text())
u=json.loads(Path('experiments/results/diagnostics/curie_data_growth_tokens_256k_p24_utility_20261005_v1.json').read_text())
assert r['status']==u['status']=='completed'
assert any(row['tag']==r['args']['tag'] and row['context_gain']>0 and row['partition_parity'] and row['matched_rng'] for row in u['rows'])
for name,digest in r['identity']['source_sha256'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
for key,arg in (('train_sha256','train_file'),('dev_sha256','dev_file')):
    h=hashlib.sha256()
    with open(r['args'][arg],'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    assert h.hexdigest()==r['identity'][key],key
CHECK
# Original256K fit1369.5s; four times data expected~5480s, allowance9000s.
MEM_CAP_KB=5000000 MEM_CAP_RSS_KB=800000 MIN_AVAIL_MB=8192 JOB_TIMEOUT_S=9000 bash experiments/queue/run_safe.sh experiments/queue/curie_data_growth_tokens_1m_b64_c16_p24_s6_20261005_v1.txt
