#!/usr/bin/env bash
set -euo pipefail
cd /workspace
for attempt in $(seq 1 50); do
  if ! tmux has-session -t curie_credit64_tokens_64k_p24_s6_20261006_v1 2>/dev/null; then break; fi
  sleep 30
done
if tmux has-session -t curie_credit64_tokens_64k_p24_s6_20261006_v1 2>/dev/null; then exit 75; fi
python3 - <<'CHECK'
import json,hashlib
from pathlib import Path
r=json.loads(Path('experiments/results/token_language/curie_credit64_tokens_64k_p24_s6_20261006_v1.json').read_text())
assert r['status']=='completed' and r['presentations_total']==131056
assert r['args']['credit_window']==64 and r['args']['chunk']==64
for name,digest in r['identity']['source_sha256'].items():
 assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
c=json.loads(Path('experiments/results/diagnostics/curie_streamed_mean_contract_64k_20261006_v1.json').read_text())
assert c['status']=='completed'
assert c['source_sha256']==hashlib.sha256(Path('experiments/streamed_token_stage_utility.py').read_bytes()).hexdigest()
CHECK
MEM_CAP_KB=5000000 MEM_CAP_RSS_KB=1200000 MIN_AVAIL_MB=8192 JOB_TIMEOUT_S=600 bash experiments/queue/run_safe.sh experiments/queue/curie_credit64_tokens_64k_p24_utility_20261006_v1.txt
python3 - <<'CHECK'
import json
from pathlib import Path
r=json.loads(Path('experiments/results/diagnostics/curie_credit64_tokens_64k_p24_utility_20261006_v1.json').read_text())
assert r['status']=='completed' and r['rows'][0]['partition_parity'] and r['rows'][0]['matched_rng']
CHECK
MEM_CAP_KB=5000000 MEM_CAP_RSS_KB=1200000 MIN_AVAIL_MB=8192 JOB_TIMEOUT_S=5400 bash experiments/queue/run_safe.sh experiments/queue/curie_credit64_tokens_64k_p24_work_20261006_v1.txt
