#!/usr/bin/env bash
set -euo pipefail
cd /workspace
export MEM_CAP_KB=8000000 MEM_CAP_RSS_KB=1500000 MIN_AVAIL_MB=9216 CGROUP_RESERVE_MB=2048 WAIT=1
JOB_TIMEOUT_S=120 bash experiments/queue/run_safe.sh experiments/queue/enabler/curie_sparse_paired_ac_v3_20261011T0727Z/contract.txt
JOB_TIMEOUT_S=120 bash experiments/queue/run_safe.sh experiments/queue/enabler/curie_sparse_paired_ac_v3_20261011T0727Z/smoke.txt
JOB_TIMEOUT_S=1200 bash experiments/queue/run_safe.sh experiments/queue/enabler/curie_sparse_paired_ac_v3_20261011T0727Z/reuse.txt
JOB_TIMEOUT_S=120 bash experiments/queue/run_safe.sh experiments/queue/enabler/curie_sparse_paired_ac_v3_20261011T0727Z/analysis.txt
JOB_TIMEOUT_S=180 bash experiments/queue/run_safe.sh experiments/queue/enabler/curie_sparse_paired_ac_v3_20261011T0727Z/diagnose.txt
