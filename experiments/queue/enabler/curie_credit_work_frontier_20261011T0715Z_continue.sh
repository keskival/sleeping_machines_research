#!/usr/bin/env bash
set -euo pipefail
cd /workspace
for attempt in $(seq 1 240); do
  if [[ -f experiments/results/credit/curie_sparse_paired_ac_diagnose_20261011T0712Z_measure.json ]]; then
    MEM_CAP_KB=8000000 MEM_CAP_RSS_KB=1500000 MIN_AVAIL_MB=9216 CGROUP_RESERVE_MB=2048 JOB_TIMEOUT_S=120 WAIT=1 bash experiments/queue/run_safe.sh experiments/queue/enabler/curie_credit_work_frontier_20261011T0715Z.txt
    exit 0
  fi
  sleep 10
done
exit 1
