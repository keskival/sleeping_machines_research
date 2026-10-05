#!/usr/bin/env bash
set -euo pipefail
cd /workspace
while tmux has-session -t curie_fas_v1_replication_20261005_v1 2>/dev/null || tmux has-session -t curie_token_lower_lr_256k_20261005_v1 2>/dev/null; do
  sleep 5
done
python3 - <<'PY'
import json
from pathlib import Path
p=Path('experiments/results/token_language/curie_data_growth_tokens_256k_b64_c16_p24_lr001_s6_20261005_v1.json')
r=json.loads(p.read_text())
assert r['status']=='completed' and r['presentations_total']==524272
PY
# Same64Kfit measured~350s/496MB. Timing wrappers preserve ordinary execution.
MEM_CAP_KB=5000000 MEM_CAP_RSS_KB=1200000 MIN_AVAIL_MB=8192 JOB_TIMEOUT_S=1200 bash experiments/queue/run_safe.sh experiments/queue/curie_data_growth_tokens_64k_p24_timing_20261005_v1.txt
