#!/usr/bin/env bash
set -euo pipefail
cd /workspace
for attempt in $(seq 1 50); do
  if ! tmux has-session -t curie_gather_64k_admission_20261006_v1 2>/dev/null; then break; fi
  sleep 30
done
if tmux has-session -t curie_gather_64k_admission_20261006_v1 2>/dev/null; then exit 75; fi
python3 - <<'CHECK'
import json
from pathlib import Path
r=json.loads(Path('experiments/results/token_language/curie_gather_tokens_64k_b64_c16_p24_s6_20261006_v1.json').read_text())
assert r['status']=='completed' and r['presentations_total']==131056
assert r['execution_override']['kind']=='Credit-window lexical gather only'
CHECK
MEM_CAP_KB=5000000 MEM_CAP_RSS_KB=800000 MIN_AVAIL_MB=8192 JOB_TIMEOUT_S=600 bash experiments/queue/run_safe.sh experiments/queue/curie_gather_tokens_64k_utility_20261006_v1.txt
# Whole audit required, not extrapolated from small-fit counts. Original64K audit took57min.
MEM_CAP_KB=5000000 MEM_CAP_RSS_KB=800000 MIN_AVAIL_MB=8192 JOB_TIMEOUT_S=5400 bash experiments/queue/run_safe.sh experiments/queue/curie_gather_tokens_64k_work_20261006_v1.txt
