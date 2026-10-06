#!/usr/bin/env bash
set -euo pipefail
cd /workspace
for attempt in $(seq 1 330); do
  if ! tmux has-session -t curie_original_1m_p32_admission_20261006_v1 2>/dev/null; then break; fi
  sleep 30
done
if tmux has-session -t curie_original_1m_p32_admission_20261006_v1 2>/dev/null; then exit 75; fi
python3 - <<'CHECK'
import hashlib,json
from pathlib import Path
r=json.loads(Path('experiments/results/token_language/curie_data_growth_tokens_1m_b64_c16_p32_s6_20261005_v1.json').read_text())
assert r['status']=='completed' and r['presentations_total']==2097136
c=json.loads(Path('experiments/results/diagnostics/curie_streamed_mean_contract_64k_20261006_v1.json').read_text())
assert c['status']=='completed'
assert c['source_sha256']==hashlib.sha256(Path('experiments/streamed_token_stage_utility.py').read_bytes()).hexdigest()
assert c['rows'][0]['mean_contract']['constant_nll_max_error']<2e-6
CHECK
# P24/1Mutility372s; P32 payload cost plus allowance1200s. Streaming memory bounded.
MEM_CAP_KB=5000000 MEM_CAP_RSS_KB=800000 MIN_AVAIL_MB=8192 JOB_TIMEOUT_S=1200 bash experiments/queue/run_safe.sh experiments/queue/curie_original_1m_p32_utility_20261006_v1.txt
