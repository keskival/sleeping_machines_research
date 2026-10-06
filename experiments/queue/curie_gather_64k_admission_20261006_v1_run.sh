#!/usr/bin/env bash
set -euo pipefail
cd /workspace
# Wait for the already admitted pair; never launch a second host job.
for attempt in $(seq 1 40); do
  if ! tmux has-session -t curie_gather_work_pair_20261006_v1 2>/dev/null; then break; fi
  sleep 30
done
if tmux has-session -t curie_gather_work_pair_20261006_v1 2>/dev/null; then exit 75; fi
python3 - <<'PY'
import json
from pathlib import Path
base=Path('experiments/results/diagnostics')
a=json.loads((base/'curie_control_tokens_8k_p24_work_20261006_v1.json').read_text())
b=json.loads((base/'curie_gather_tokens_8k_p24_work_20261006_v1.json').read_text())
for d in (a,b):
    assert d['status']=='completed' and d['curve_parity_max_error']<2e-6
    assert d['fitting_targets']==16368 and d['optimizer_updates']==32
    for phase in ('fitting','inference'):
        assert d[phase]['formula_coverage_complete'] and not d[phase]['unsupported_floating_operators']
assert b['fitting']['arithmetic_flops']<a['fitting']['arithmetic_flops']
assert abs(b['selected_dev_nll']-a['selected_dev_nll'])<.01
print('Completed own-parent parity, full arithmetic coverage, lower fitting work and small-fit quality gate passed.',flush=True)
PY
# Budget based on 64K original ordinary fit263s and measured gather fit savings.
MEM_CAP_KB=5000000 MEM_CAP_RSS_KB=800000 MIN_AVAIL_MB=8192 JOB_TIMEOUT_S=1200 bash experiments/queue/run_safe.sh experiments/queue/curie_gather_tokens_64k_b64_c16_p24_s6_20261006_v1.txt
