#!/usr/bin/env bash
set -euo pipefail
cd /workspace
export MEM_CAP_KB=8000000 MEM_CAP_RSS_KB=1500000 MIN_AVAIL_MB=9216 CGROUP_RESERVE_MB=2048 WAIT=1
JOB_TIMEOUT_S=1200 bash experiments/queue/run_safe.sh experiments/queue/enabler/curie_sparse_paired_ac_v2_20261011T0930Z/acquire.txt
JOB_TIMEOUT_S=120 bash experiments/queue/run_safe.sh experiments/queue/enabler/curie_sparse_paired_ac_v2_20261011T0930Z/analyze_acquire.txt
python3 - <<'GATE'
import json
p=json.load(open('experiments/results/credit/curie_sparse_paired_ac_v2_20261011T0930Z_analyze_acquire.json'))
assert p['status']=='completed' and p['continue_registered_ladder']
GATE
JOB_TIMEOUT_S=1800 bash experiments/queue/run_safe.sh experiments/queue/enabler/curie_sparse_paired_ac_v2_20261011T0930Z/reuse.txt
JOB_TIMEOUT_S=120 bash experiments/queue/run_safe.sh experiments/queue/enabler/curie_sparse_paired_ac_v2_20261011T0930Z/analyze_reuse.txt
