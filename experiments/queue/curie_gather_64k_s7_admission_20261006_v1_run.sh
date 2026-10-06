#!/usr/bin/env bash
set -euo pipefail
cd /workspace
for attempt in $(seq 1 220); do
  if ! tmux has-session -t curie_gather_64k_followup_20261006_v1 2>/dev/null; then break; fi
  sleep 30
done
if tmux has-session -t curie_gather_64k_followup_20261006_v1 2>/dev/null; then exit 75; fi
python3 - <<'CHECK'
import json
from pathlib import Path
p=Path('experiments/results/diagnostics/curie_gather_tokens_64k_work_20261006_v1.json')
r=json.loads(p.read_text())
assert r['status']=='completed' and r['curve_parity_max_error']<2e-6
assert r['fitting']['formula_coverage_complete'] and r['inference']['formula_coverage_complete']
assert r['fitting_targets']==131056
CHECK
MEM_CAP_KB=5000000 MEM_CAP_RSS_KB=800000 MIN_AVAIL_MB=8192 JOB_TIMEOUT_S=1200 bash experiments/queue/run_safe.sh experiments/queue/curie_gather_tokens_64k_b64_c16_p24_s7_20261006_v1.txt
