#!/usr/bin/env bash
set -euo pipefail
cd /workspace
for attempt in $(seq 1 200); do
 if ! tmux has-session -t curie_credit64_tokens_64k_p24_followthrough_20261006_v1 2>/dev/null; then break; fi
 sleep 30
done
if tmux has-session -t curie_credit64_tokens_64k_p24_followthrough_20261006_v1 2>/dev/null; then exit 75; fi
python3 - <<'CHECK'
import json
from pathlib import Path
r=json.loads(Path('experiments/results/diagnostics/curie_credit64_tokens_64k_p24_work_20261006_v1.json').read_text())
assert r['status']=='completed' and r['optimizer_updates']==256 and r['curve_parity_max_error']<2e-6
assert all(r[k]['formula_coverage_complete'] for k in ('fitting','inference'))
CHECK
MEM_CAP_KB=5000000 MEM_CAP_RSS_KB=1200000 MIN_AVAIL_MB=8192 JOB_TIMEOUT_S=120 bash experiments/queue/run_safe.sh experiments/queue/curie_split_horizon_state_contracts_20261006_v1.txt
MEM_CAP_KB=5000000 MEM_CAP_RSS_KB=1200000 MIN_AVAIL_MB=8192 JOB_TIMEOUT_S=600 bash experiments/queue/run_safe.sh experiments/queue/curie_split_horizon_driver_contracts_20261006_v1.txt
