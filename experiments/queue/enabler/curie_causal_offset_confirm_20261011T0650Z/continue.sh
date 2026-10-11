#!/usr/bin/env bash
set -euo pipefail
cd /workspace
export MEM_CAP_KB=8000000 MEM_CAP_RSS_KB=1500000 MIN_AVAIL_MB=9216 CGROUP_RESERVE_MB=2048 JOB_TIMEOUT_S=600 WAIT=1
bash experiments/queue/run_safe.sh experiments/queue/enabler/curie_causal_offset_confirm_20261011T0650Z/stream473.txt
bash experiments/queue/run_safe.sh experiments/queue/enabler/curie_causal_offset_confirm_20261011T0650Z/stream474.txt
bash experiments/queue/run_safe.sh experiments/queue/enabler/curie_causal_offset_confirm_20261011T0650Z/decision.txt
