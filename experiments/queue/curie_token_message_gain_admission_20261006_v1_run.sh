#!/usr/bin/env bash
set -euo pipefail
cd /workspace
for attempt in $(seq 1 270); do
  if ! tmux has-session -t curie_gather_64k_s7_admission_20261006_v1 2>/dev/null; then break; fi
  sleep 30
done
if tmux has-session -t curie_gather_64k_s7_admission_20261006_v1 2>/dev/null; then exit 75; fi
python3 - <<'CHECK'
import json
from pathlib import Path
for seed in (6,7):
    r=json.loads(Path(f'experiments/results/token_language/curie_gather_tokens_64k_b64_c16_p24_s{seed}_20261006_v1.json').read_text())
    assert r['status']=='completed' and r['presentations_total']==131056
CHECK
MEM_CAP_KB=5000000 MEM_CAP_RSS_KB=800000 MIN_AVAIL_MB=8192 JOB_TIMEOUT_S=600 bash experiments/queue/run_safe.sh experiments/queue/curie_token_message_gain_pair_20261006_v1.txt
